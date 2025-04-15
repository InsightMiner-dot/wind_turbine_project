import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Load the dataset
df = pd.read_csv('wind_turbine_maintenance_data_with_id.csv')

# Features and target
X = df.drop(['failure', 'Turbine_ID'], axis=1)  # Exclude Turbine_ID and target
y = df['failure']

# Split into train (70%), validation (15%), and test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define models and hyperparameter grids
models = {
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier()
}

param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20]
    },
    # 'SVM': {
    #     'C': [0.1, 1, 10],
    #     'kernel': ['linear', 'rbf']
    # },
    'KNN': {
        'n_neighbors': [5, 10],
        'weights': ['uniform', 'distance']
    }
}

# Perform Grid Search for all models except SVM
best_models = {}
for model_name in models:
    if model_name != 'SVM':
        grid_search = GridSearchCV(models[model_name], param_grids[model_name], scoring='roc_auc', cv=5)
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    else:
        # Directly fit the SVM model
        models[model_name].fit(X_train, y_train)
        best_models[model_name] = models[model_name]

# Evaluate models
results = {}
for model_name, model in best_models.items():
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    results[model_name] = {
        'Train Accuracy': accuracy_score(y_train, y_train_pred),
        'Train Precision': precision_score(y_train, y_train_pred, zero_division=0),
        'Train Recall': recall_score(y_train, y_train_pred),
        'Train F1-Score': f1_score(y_train, y_train_pred),
        'Train ROC-AUC': roc_auc_score(y_train, y_train_prob),
        'Validation Accuracy': accuracy_score(y_val, y_val_pred),
        'Validation Precision': precision_score(y_val, y_val_pred, zero_division=0),
        'Validation Recall': recall_score(y_val, y_val_pred),
        'Validation F1-Score': f1_score(y_val, y_val_pred),
        'Validation ROC-AUC': roc_auc_score(y_val, y_val_prob),
        'Test Accuracy': accuracy_score(y_test, y_test_pred),
        'Test Precision': precision_score(y_test, y_test_pred, zero_division=0),
        'Test Recall': recall_score(y_test, y_test_pred),
        'Test F1-Score': f1_score(y_test, y_test_pred),
        'Test ROC-AUC': roc_auc_score(y_test, y_test_prob)
    }

results_df = pd.DataFrame(results).T
print(results_df)

# Select the best model based on Validation ROC-AUC
best_model_name = results_df['Validation ROC-AUC'].idxmax()
best_model = best_models[best_model_name]

# Fit the best model to the training data
best_model.fit(X_train, y_train)

# Evaluate test set
test_data = df.iloc[X_test.index].copy()
test_data['predictions'] = best_model.predict(X_test)
test_data['probabilities'] = best_model.predict_proba(X_test)[:, 1]

# Save test results to Excel
test_data.to_excel('test_results.xlsx', index=False)

# Save the model in pkl
joblib.dump(best_model, 'best_model.pkl')

print(f"Best model ({best_model_name}) saved as 'best_model.pkl' and test results saved to 'test_results.xlsx'.")