import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go

# Load the saved model
model = joblib.load(r'D:\2024\streamlit\best_model.pkl')

# Title of the app
st.title("Wind Turbine Predictive Maintenance - Live Prediction Demo")

# Sidebar for navigation
st.sidebar.header("Navigation")
option = st.sidebar.radio("Choose an option", ["Interactive Input", "Batch Prediction", "Live Prediction"])

# Function to get user inputs for interactive mode
def get_user_input():
    st.sidebar.header("Input Features")
    turbine_id = st.sidebar.selectbox('Turbine ID', [f'WTG{i+1}' for i in range(10)])  # Select from WTG1 to WTG10
    rotor_speed = st.sidebar.slider('Rotor Speed (RPM)', 10.0, 20.0, 15.0)
    generator_speed = st.sidebar.slider('Generator Speed (RPM)', 1400.0, 1600.0, 1500.0)
    power_output = st.sidebar.slider('Power Output (kW)', 1500.0, 2500.0, 2000.0)
    blade_pitch_angle = st.sidebar.slider('Blade Pitch Angle (degrees)', 0.0, 10.0, 5.0)
    gearbox_temp = st.sidebar.slider('Gearbox Temperature (°C)', 50.0, 100.0, 70.0)
    generator_temp = st.sidebar.slider('Generator Temperature (°C)', 60.0, 110.0, 80.0)
    vibration = st.sidebar.slider('Vibration (mm/s)', 1.0, 5.0, 2.0)
    oil_pressure = st.sidebar.slider('Oil Pressure (bar)', 80.0, 120.0, 100.0)
    wind_speed = st.sidebar.slider('Wind Speed (m/s)', 5.0, 15.0, 10.0)
    ambient_temp = st.sidebar.slider('Ambient Temperature (°C)', 10.0, 30.0, 20.0)
    humidity = st.sidebar.slider('Humidity (%)', 40.0, 80.0, 60.0)
    time_since_maintenance = st.sidebar.slider('Time Since Last Maintenance (days)', 0, 365, 180)

    # Store inputs in a dictionary
    user_data = {
        'Turbine_ID': turbine_id,
        'rotor_speed': rotor_speed,
        'generator_speed': generator_speed,
        'power_output': power_output,
        'blade_pitch_angle': blade_pitch_angle,
        'gearbox_temp': gearbox_temp,
        'generator_temp': generator_temp,
        'vibration': vibration,
        'oil_pressure': oil_pressure,
        'wind_speed': wind_speed,
        'ambient_temp': ambient_temp,
        'humidity': humidity,
        'time_since_maintenance': time_since_maintenance
    }

    # Convert to DataFrame
    features = pd.DataFrame(user_data, index=[0])
    return features

# Function to process uploaded file and make predictions
def process_uploaded_file(uploaded_file):
    # Read the file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None

    # Check if the required columns are present
    required_columns = [
        'Turbine_ID', 'rotor_speed', 'generator_speed', 'power_output', 'blade_pitch_angle',
        'gearbox_temp', 'generator_temp', 'vibration', 'oil_pressure',
        'wind_speed', 'ambient_temp', 'humidity', 'time_since_maintenance'
    ]
    if not all(col in df.columns for col in required_columns):
        st.error(f"The uploaded file must contain the following columns: {required_columns}")
        return None

    # Make predictions
    df['Failure_Probability'] = model.predict_proba(df[required_columns[1:]])[:, 1]  # Exclude Turbine_ID for prediction
    df['Warning'] = (df['Failure_Probability'] > 0.7).astype(int)  # 1 = Warning, 0 = No Warning

    return df

# Function to simulate live data
def generate_live_data():
    # Generate random data within specified ranges
    live_data = {
        'Turbine_ID': np.random.choice([f'WTG{i+1}' for i in range(6)]),  # Random Turbine ID
        'rotor_speed': np.random.uniform(12.42800059, 16.25015359),
        'generator_speed': np.random.uniform(1430.549684, 1661.231214),
        'power_output': np.random.uniform(1535.832802, 2515.135746),
        'blade_pitch_angle': np.random.uniform(3.713305598, 7.217959567),
        'gearbox_temp': np.random.uniform(51.78456692, 94.35799204),
        'generator_temp': np.random.uniform(60.51528976, 111.8489284),
        'vibration': np.random.uniform(1.388468529, 3.312948245),
        'oil_pressure': np.random.uniform(92.17177633, 113.6196584),
        'wind_speed': np.random.uniform(4.646878697, 15.51269193),
        'ambient_temp': np.random.uniform(15.4430759, 27.02471214),
        'humidity': np.random.uniform(51.30045461, 67.54600845),
        'time_since_maintenance': np.random.randint(14, 330)
    }
    return pd.DataFrame(live_data, index=[0])

# Interactive Input Mode
if option == "Interactive Input":
    st.header("Interactive Input Mode")
    user_input = get_user_input()

    # Display user inputs
    st.subheader("User Input Features")
    st.write(user_input)

    # Predict failure probability
    prediction = model.predict_proba(user_input.drop(columns=['Turbine_ID']))[:, 1][0]
    failure_probability = round(prediction * 100, 2)

    # Set warning threshold
    warning_threshold = .5

    # Display prediction and warning
    st.subheader("Prediction")
    st.write(f"Probability of Failure: {failure_probability}%")

    if prediction > warning_threshold:
        st.error("WARNING: High probability of failure. Schedule maintenance immediately!")
    else:
        st.success("No immediate risk. No maintenance required.")

# Batch Prediction Mode
elif option == "Batch Prediction":
    st.header("Batch Prediction Mode")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Process the uploaded file
        df = process_uploaded_file(uploaded_file)

        if df is not None:
            # Display the processed data
            st.subheader("Processed Data with Predictions")
            st.write(df)

            # Download the processed file
            st.subheader("Download Predictions")
            output_file = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=output_file,
                file_name='predictions.csv',
                mime='text/csv'
            )

# Live Prediction Mode
elif option == "Live Prediction":
    st.header("Live Prediction Mode")
    st.write("Simulating live data streaming and real-time predictions...")

    # Placeholder for live data and predictions
    live_data_placeholder = st.empty()
    prediction_placeholder = st.empty()
    warning_placeholder = st.empty()
    chart_placeholder = st.empty()

    # DataFrame to store live data for plotting
    live_data_df = pd.DataFrame(columns=[
        'rotor_speed', 'generator_speed', 'power_output', 'blade_pitch_angle',
        'gearbox_temp', 'generator_temp', 'vibration', 'oil_pressure',
        'wind_speed', 'ambient_temp', 'humidity', 'time_since_maintenance', 'Failure_Probability'
    ])

    # Simulate live data streaming
    for _ in range(20):  # Simulate 20 data points
        # Generate live data
        live_data = generate_live_data()

        # Predict failure probability
        prediction = model.predict_proba(live_data.drop(columns=['Turbine_ID']))[:, 1][0]
        failure_probability = round(prediction * 100, 2)
        live_data['Failure_Probability'] = failure_probability

        # Append live data to DataFrame
        live_data_df = pd.concat([live_data_df, live_data], ignore_index=True)

        # Display live data
        live_data_placeholder.subheader("Live Sensor Data")
        live_data_placeholder.write(live_data)

        # Display prediction
        prediction_placeholder.subheader("Prediction")
        prediction_placeholder.write(f"Probability of Failure: {failure_probability}%")

        # Set warning threshold
        warning_threshold = 0.5

        # Display warning
        if prediction > warning_threshold:
            warning_placeholder.error("WARNING: High probability of failure. Schedule maintenance immediately!")
        else:
            warning_placeholder.success("No immediate risk. No maintenance required.")

        # Plot the live data using Plotly
        fig = go.Figure()

        # Add traces for rotor_speed, generator_speed, and power_output
        fig.add_trace(go.Scatter(x=live_data_df.index, y=live_data_df['rotor_speed'], mode='lines', name='Rotor Speed'))
        fig.add_trace(go.Scatter(x=live_data_df.index, y=live_data_df['generator_speed'], mode='lines', name='Generator Speed'))
        fig.add_trace(go.Scatter(x=live_data_df.index, y=live_data_df['power_output'], mode='lines', name='Power Output'))

        # Add a secondary y-axis for Failure Probability
        fig.add_trace(go.Scatter(x=live_data_df.index, y=live_data_df['Failure_Probability'], mode='lines', name='Failure Probability', yaxis='y2'))

        # Update layout for dual y-axis
        fig.update_layout(
            yaxis=dict(title='Rotor Speed / Generator Speed / Power Output'),
            yaxis2=dict(title='Failure Probability', overlaying='y', side='right')
        )

        # Display the plot
        chart_placeholder.plotly_chart(fig)

        # Wait for 2 seconds before the next data point
        time.sleep(2)