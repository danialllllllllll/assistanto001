from flask import Flask, jsonify, render_template
from flask_cors import CORS
import json
import os
from datetime import datetime
import threading

app = Flask(__name__)
CORS(app)

training_state = {
    'current_stage': 'Initializing',
    'stage_index': 0,
    'understanding_score': 0.0,
    'confidence': 0.0,
    'accuracy': 0.0,
    'iteration': 0,
    'total_stages': 7,
    'progress_percent': 0.0,
    'personality_traits': {},
    'philosophy_insights': [],
    'core_values_status': [],
    'training_active': False,
    'stage_passed': False,
    'history': []
}

def update_training_state(stage_name, stage_index, understanding, confidence, accuracy, iteration, total_stages):
    """Update the global training state"""
    global training_state
    training_state['current_stage'] = stage_name
    training_state['stage_index'] = stage_index
    training_state['understanding_score'] = float(understanding)
    training_state['confidence'] = float(confidence)
    training_state['accuracy'] = float(accuracy)
    training_state['iteration'] = int(iteration)
    training_state['total_stages'] = int(total_stages)
    training_state['progress_percent'] = (stage_index / total_stages) * 100
    training_state['training_active'] = True

def update_personality(traits):
    """Update personality traits"""
    global training_state
    training_state['personality_traits'] = traits

def add_philosophy_insight(insight):
    """Add a philosophy insight"""
    global training_state
    training_state['philosophy_insights'].append({
        'timestamp': datetime.now().isoformat(),
        'insight': insight
    })
    if len(training_state['philosophy_insights']) > 10:
        training_state['philosophy_insights'] = training_state['philosophy_insights'][-10:]

def update_core_values(values_status):
    """Update core values status"""
    global training_state
    training_state['core_values_status'] = values_status

def add_to_history(stage_result):
    """Add stage result to history"""
    global training_state
    training_state['history'].append(stage_result)

@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current training status"""
    return jsonify({
        'current_stage': training_state['current_stage'],
        'understanding_score': training_state['understanding_score'],
        'confidence': training_state['confidence'],
        'accuracy': training_state['accuracy'],
        'iteration': training_state['iteration'],
        'stage_index': training_state['stage_index'],
        'total_stages': training_state['total_stages'],
        'training_active': training_state['training_active']
    })

@app.route('/api/progress')
def get_progress():
    """Get full training progress data"""
    return jsonify({
        'current_stage': training_state['current_stage'],
        'stage_index': training_state['stage_index'],
        'total_stages': training_state['total_stages'],
        'progress_percent': training_state['progress_percent'],
        'understanding_score': training_state['understanding_score'],
        'confidence': training_state['confidence'],
        'accuracy': training_state['accuracy'],
        'iteration': training_state['iteration'],
        'history': training_state['history'],
        'training_active': training_state['training_active']
    })

@app.route('/api/personality')
def get_personality():
    """Get current personality traits"""
    return jsonify({
        'traits': training_state['personality_traits'],
        'stage': training_state['current_stage']
    })

@app.route('/api/philosophy')
def get_philosophy():
    """Get philosophy insights"""
    return jsonify({
        'insights': training_state['philosophy_insights'],
        'stage': training_state['current_stage']
    })

@app.route('/api/core_values')
def get_core_values():
    """Get core values status"""
    return jsonify({
        'values': training_state['core_values_status'],
        'enforced': True,
        'immutable': True
    })

def run_flask_app():
    """Run Flask app in a thread"""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def start_flask_background():
    """Start Flask in background thread"""
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    print("Flask web interface started at http://0.0.0.0:5000")
    return flask_thread

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
