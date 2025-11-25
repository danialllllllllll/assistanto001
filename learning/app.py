from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import json
import os
from datetime import datetime
import threading
import time
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__, template_folder='../utils')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Production-ready configuration
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request

# Import and register AI API blueprint
from learning.ai_api import ai_api, update_ai_state
app.register_blueprint(ai_api)

# Global progress estimator for background ETA
global_progress_estimator = None

def background_eta_calculator():
    """Background thread to calculate ETAs without blocking training"""
    while True:
        try:
            if global_progress_estimator and training_state.get('training_active'):
                stage_eta = global_progress_estimator.estimate_current_stage_completion()
                total_eta = global_progress_estimator.estimate_total_completion()
                
                if stage_eta:
                    training_state['time_estimate']['next_stage_eta'] = stage_eta.get('eta_formatted', 'Calculating...')
                    training_state['time_estimate']['next_stage_eta_seconds'] = stage_eta.get('eta_seconds', None)
                    training_state['time_estimate']['stage_confidence'] = stage_eta.get('confidence', 0.5)
                    training_state['time_estimate']['stage_method'] = stage_eta.get('method', 'unknown')
                else:
                    training_state['time_estimate']['next_stage_eta'] = 'Analyzing...'
                    training_state['time_estimate']['stage_confidence'] = 0.0
                
                if total_eta:
                    training_state['time_estimate']['total_completion_eta'] = total_eta.get('eta_formatted', 'Unknown')
                    training_state['time_estimate']['total_completion_eta_seconds'] = total_eta.get('eta_seconds', None)
                    training_state['time_estimate']['total_confidence'] = total_eta.get('confidence', 0.5)
                else:
                    training_state['time_estimate']['total_completion_eta'] = 'Analyzing...'
                    training_state['time_estimate']['total_confidence'] = 0.0
        except Exception as e:
            print(f"Background ETA error: {e}")
            import traceback
            traceback.print_exc()
        
        time.sleep(1)  # Update every second for more responsive UI

# Start background ETA calculator
eta_thread = threading.Thread(target=background_eta_calculator, daemon=True)
eta_thread.start()

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
    'last_understanding_update': None,
    'generation': 0,
    'mutations': [],
    'nodes_created': 0,
    'nodes_pruned': 0,
    'evolution_events': []
}

def set_progress_estimator(estimator):
    """Register the progress estimator for background ETA calculation"""
    global global_progress_estimator
    global_progress_estimator = estimator

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

def update_evolution(generation, mutations, nodes_created, nodes_pruned, evolution_events):
    """Update evolution and mutation data"""
    global training_state
    training_state['generation'] = generation
    training_state['mutations'] = mutations[-20:]  # Keep last 20
    training_state['nodes_created'] = nodes_created
    training_state['nodes_pruned'] = nodes_pruned
    training_state['evolution_events'] = evolution_events[-20:]  # Keep last 20

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

@app.route('/health')
def health_check():
    """Health check endpoint for deployment monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'training_active': training_state.get('training_active', False),
        'current_stage': training_state.get('current_stage', 'Unknown')
    })

@app.route('/api/metrics')
def get_metrics():
    """Comprehensive metrics endpoint"""
    return jsonify({
        'training': {
            'stage': training_state.get('current_stage'),
            'iteration': training_state.get('iteration'),
            'understanding': training_state.get('understanding_score'),
            'accuracy': training_state.get('accuracy')
        },
        'system': {
            'active': training_state.get('training_active'),
            'generation': training_state.get('generation', 0),
            'mutations': len(training_state.get('mutations', []))
        },
        'performance': {
            'nodes_created': training_state.get('nodes_created', 0),
            'nodes_pruned': training_state.get('nodes_pruned', 0)
        }
    })

@app.route('/api/web_knowledge')
def get_web_knowledge():
    """Get web-acquired knowledge"""
    return jsonify({
        'recent_knowledge': web_knowledge_state['recent_knowledge'],
        'stats': web_knowledge_state['stats'],
        'active': True
    })

@app.route('/api/evolution')
def get_evolution():
    """Get evolution and mutation data"""
    return jsonify({
        'generation': training_state.get('generation', 0),
        'mutations': training_state.get('mutations', []),
        'nodes_created': training_state.get('nodes_created', 0),
        'nodes_pruned': training_state.get('nodes_pruned', 0),
        'evolution_events': training_state.get('evolution_events', [])
    })

@app.route('/api/network_state')
def get_network_state():
    """Get current neural network structure and activation states with mutations"""
    global training_state
    
    try:
        stage_index = training_state.get('stage_index', 0)
        understanding = training_state.get('understanding_score', 0)
        iteration = training_state.get('iteration', 0)
        
        # Dynamic layer sizes based on stage
        if stage_index == 0:  # Baby Steps
            layer_sizes = [32, 48, 24, 4]
        elif stage_index == 1:  # Toddler
            layer_sizes = [48, 72, 48, 8]
        elif stage_index == 2:  # Pre-K
            layer_sizes = [64, 96, 64, 10]
        elif stage_index == 3:  # Elementary
            layer_sizes = [80, 120, 80, 12]
        elif stage_index == 4:  # Teen
            layer_sizes = [96, 144, 96, 16]
        elif stage_index == 5:  # Scholar
            layer_sizes = [112, 168, 112, 20]
        else:  # Thinker
            layer_sizes = [128, 192, 128, 24]
        
        nodes = []
        connections = []
        node_id = 0
        
        # More dynamic activation
        base_activation = min(0.95, understanding + 0.05)
        
        # Build nodes with evolutionary variance
        for layer_idx, size in enumerate(layer_sizes):
            for node_idx in range(size):
                # Time-based evolution
                evolution_factor = (iteration % 100) / 100.0
                noise = (node_idx * 0.13 + layer_idx * 0.07 + evolution_factor * 0.1) % 0.4
                activation = base_activation + noise - 0.2
                activation = max(0.05, min(0.98, activation))
                
                # Some nodes randomly die/spawn based on iteration
                is_active = activation > 0.3 and (node_idx + iteration) % 7 != 0
                
                nodes.append({
                    'id': node_id,
                    'layer': layer_idx,
                    'activation': float(activation),
                    'active': is_active
                })
                node_id += 1
        
        # Build connections with evolution
        node_offset = 0
        for layer_idx in range(len(layer_sizes) - 1):
            layer_size = layer_sizes[layer_idx]
            next_layer_size = layer_sizes[layer_idx + 1]
            
            connection_density = 0.4  # Show 40% of connections
            
            for i in range(layer_size):
                for j in range(next_layer_size):
                    if (i * j + layer_idx + iteration // 10) % int(1/connection_density) == 0:
                        weight = base_activation * ((i + j + iteration // 50) % 10) / 10
                        connections.append({
                            'from': node_offset + i,
                            'to': node_offset + layer_size + j,
                            'weight': float(weight),
                            'active': weight > 0.25
                        })
            
            node_offset += layer_size
        
        return jsonify({
            'nodes': nodes,
            'connections': connections,
            'layer_sizes': layer_sizes,
            'iteration': iteration,
            'evolution_factor': float((iteration % 100) / 100.0)
        })
        
    except Exception as e:
        print(f"Network state error: {e}")
        return jsonify({'nodes': [], 'connections': [], 'layer_sizes': []})

def run_flask_app():
    """Run Flask app in production mode"""
    try:
        from waitress import serve
        print("Starting production WSGI server on http://0.0.0.0:5000...")
        serve(app, host='0.0.0.0', port=5000, threads=4, channel_timeout=120)
    except ImportError:
        print("Waitress not found, falling back to Flask development server...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

def start_flask_background():
    """Start Flask in background thread"""
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    print("Flask web interface started at http://0.0.0.0:5000")
    return flask_thread

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)