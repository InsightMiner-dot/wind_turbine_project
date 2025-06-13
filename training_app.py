from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import os
import threading
import mlflow
from mlflow.tracking import MlflowClient
from train_module import run_training
import atexit
import shutil
from datetime import datetime, timedelta
app = Flask(__name__)
socketio = SocketIO(app)

# Training status tracking
training_status = {
    'current_step': '',
    'details': '',
    'epoch_progress': '',
    'dataset_info': ''
}

# Add to your global variables
training_start_time = None

#the update_training_status function
def update_training_status(step, details=None, epoch_progress=None, dataset_info=None):
    global training_status
    training_status['current_step'] = step
    if details:
        training_status['details'] = details
    if epoch_progress:
        training_status['epoch_progress'] = epoch_progress
    if dataset_info:
        training_status['dataset_info'] = dataset_info
    # Calculate elapsed time if training has started
    if training_start_time:
        elapsed = datetime.now() - training_start_time
        training_status['elapsed_time'] = str(elapsed).split('.')[0]  # Remove microseconds
    
    socketio.emit('training_update', training_status)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/train', methods=['POST'])
def train_model():
    folder_path = request.form.get('folder_path', '').strip()
    if not folder_path:
        return jsonify({'error': 'No folder path provided'}), 400

    if not os.path.isdir(folder_path):
        return jsonify({'error': 'Invalid folder path'}), 400

    # Check dataset structure
    train_path = os.path.join(folder_path, 'train')
    val_path = os.path.join(folder_path, 'val')
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        return jsonify({'error': 'Dataset must contain train/ and val/ subdirectories'}), 400

    params = {
        'model_name': request.form.get('model_name', 'efficientnet_b0'),
        'optimizer': request.form.get('optimizer', 'Adam'),
        'learning_rate': float(request.form.get('learning_rate', 0.001)),
        'num_epochs': int(request.form.get('num_epochs', 5)),
        'batch_size': int(request.form.get('batch_size', 4)),
        'data_dir': folder_path,
        'status_updater': update_training_status
    }

    thread = threading.Thread(target=run_training, kwargs=params)
    thread.start()

    return jsonify({'message': 'Training started successfully'}), 200


@app.route('/training_status')
def get_training_status():
    return jsonify(training_status)

@app.route('/experiments')
def get_experiments():
    experiments = []
    try:
        client = MlflowClient()
        for exp in client.search_experiments():
            runs = client.search_runs(exp.experiment_id)
            for run in runs:
                experiments.append({
                    'experiment_id': exp.experiment_id,
                    'run_id': run.info.run_id,
                    'name': exp.name,
                    'status': run.info.status,
                    'metrics': run.data.metrics,
                    'params': run.data.params,
                    'start_time': run.info.start_time
                })
    except Exception as e:
        print(f"Error fetching experiments: {e}")
    
    return jsonify(experiments)

if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=False)