from flask import Flask, jsonify, render_template
from flask_cors import CORS
import json
import os
from datetime import datetime
import threading
import time

app = Flask(__name__)
CORS(app)

# Import and register AI API blueprint
from interfaces.ai_api import ai_api, update_ai_state
app.register_blueprint(ai_api)

web_knowledge_state = {
    'recent_knowledge': [],
    'stats': {
        'total_acquired': 0,
        'by_stage': {},
        'sources_count': 0
    }
}

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
    'history': [],
    'time_estimate': {
        'next_stage_eta': 'Calculating...',
        'next_stage_eta_seconds': None,
        'stage_confidence': 0.0,
        'total_completion_eta': 'Calculating...',
        'total_completion_eta_seconds': None,
        'total_confidence': 0.0
    },
    'last_iteration_time': None,
    'last_understanding_update': None
}

def update_training_state(stage_name, stage_index, understanding, confidence, accuracy, iteration, total_stages, stage_eta=None, total_eta=None):
    """Update the global training state with functional time estimation"""
    global training_state

    current_time = time.time()

    training_state['current_stage'] = stage_name
    training_state['stage_index'] = stage_index
    training_state['understanding_score'] = float(understanding)
    training_state['confidence'] = float(confidence)
    training_state['accuracy'] = float(accuracy)
    training_state['iteration'] = int(iteration)
    training_state['total_stages'] = int(total_stages)
    training_state['progress_percent'] = (stage_index / total_stages) * 100
    training_state['training_active'] = True
    training_state['last_understanding_update'] = current_time

    # Update time estimates from progress estimator
    if stage_eta:
        training_state['time_estimate']['next_stage_eta'] = stage_eta.get('eta_formatted', 'Calculating...')
        training_state['time_estimate']['next_stage_eta_seconds'] = stage_eta.get('eta_seconds', None)
        training_state['time_estimate']['stage_confidence'] = stage_eta.get('confidence', 0.5)

    if total_eta:
        training_state['time_estimate']['total_completion_eta'] = total_eta.get('eta_formatted', 'Unknown')
        training_state['time_estimate']['total_completion_eta_seconds'] = total_eta.get('eta_seconds', None)
        training_state['time_estimate']['total_confidence'] = total_eta.get('confidence', 0.5)

    training_state['last_iteration_time'] = current_time

def _format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds is None or seconds <= 0:
        return '0s'

    seconds = int(seconds)
    parts = []

    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")

    return ' '.join(parts)

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

def update_web_knowledge(knowledge_items, stats):
    """Update web knowledge state"""
    global web_knowledge_state
    web_knowledge_state['recent_knowledge'] = knowledge_items
    web_knowledge_state['stats'] = stats

@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current training status with time estimates"""
    return jsonify({
        'current_stage': training_state['current_stage'],
        'understanding_score': training_state['understanding_score'],
        'confidence': training_state['confidence'],
        'accuracy': training_state['accuracy'],
        'iteration': training_state['iteration'],
        'stage_index': training_state['stage_index'],
        'total_stages': training_state['total_stages'],
        'training_active': training_state['training_active'],
        'time_estimate': training_state['time_estimate']
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

@app.route('/api/web_knowledge')
def get_web_knowledge():
    """Get web-acquired knowledge"""
    return jsonify({
        'recent_knowledge': web_knowledge_state['recent_knowledge'],
        'stats': web_knowledge_state['stats'],
        'active': True
    })

@app.route('/api/network_state')
def get_network_state():
    """Get current neural network structure and activation states"""
    global training_state
    
    try:
        # Create a simplified network visualization
        stage_index = training_state.get('stage_index', 0)
        understanding = training_state.get('understanding_score', 0)
        
        # Define layer sizes for visualization
        layer_sizes = [64, 128, 64, 10]  # input, hidden1, hidden2, output
        nodes = []
        connections = []
        node_id = 0
        
        # Calculate activation based on training progress
        base_activation = min(0.9, understanding)
        
        # Build nodes
        for layer_idx, size in enumerate(layer_sizes):
            for node_idx in range(size):
                # Vary activation based on position and training state
                noise = (node_idx * 0.1 + layer_idx * 0.05) % 0.3
                activation = base_activation + noise - 0.15
                activation = max(0.1, min(0.95, activation))
                
                nodes.append({
                    'id': node_id,
                    'layer': layer_idx,
                    'activation': float(activation),
                    'active': activation > 0.4
                })
                node_id += 1
        
        # Build connections with weights
        node_offset = 0
        for layer_idx in range(len(layer_sizes) - 1):
            layer_size = layer_sizes[layer_idx]
            next_layer_size = layer_sizes[layer_idx + 1]
            
            # Sample connections (not all, to keep visualization clean)
            connection_density = 0.3  # Show 30% of connections
            
            for i in range(layer_size):
                for j in range(next_layer_size):
                    if (i * j + layer_idx) % int(1/connection_density) == 0:
                        # Generate weight based on training progress
                        weight = base_activation * ((i + j) % 10) / 10
                        connections.append({
                            'from': node_offset + i,
                            'to': node_offset + layer_size + j,
                            'weight': float(weight),
                            'active': weight > 0.2
                        })
            
            node_offset += layer_size
        
        return jsonify({
            'nodes': nodes,
            'connections': connections,
            'layer_sizes': layer_sizes
        })
        
    except Exception as e:
        print(f"Network state error: {e}")
        return jsonify({'nodes': [], 'connections': [], 'layer_sizes': []})

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