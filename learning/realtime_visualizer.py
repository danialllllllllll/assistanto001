"""
Real-Time Neural Processing Visualizer
Shows active neurons, connection weights, and processing flow as Whimsy learns
Integrates with genetic learning to display human-like adaptive patterns
"""

import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import deque

class NeuronState:
    """Tracks the state of an individual neuron"""
    def __init__(self, layer: int, index: int):
        self.layer = layer
        self.index = index
        self.activation_history = deque(maxlen=100)
        self.gradient_history = deque(maxlen=100)
        self.connections_in = []
        self.connections_out = []
        self.specialization = None
        self.importance_score = 0.0
        self.age = 0
    
    def update_activation(self, activation: float):
        """Update neuron activation"""
        self.activation_history.append({
            'value': float(activation),
            'timestamp': datetime.now().isoformat()
        })
        self.age += 1
    
    def get_current_activation(self) -> float:
        """Get most recent activation"""
        if self.activation_history:
            return self.activation_history[-1]['value']
        return 0.0
    
    def get_avg_activation(self) -> float:
        """Get average activation over history"""
        if not self.activation_history:
            return 0.0
        return np.mean([h['value'] for h in self.activation_history])
    
    def calculate_importance(self) -> float:
        """Calculate neuron importance based on activation patterns"""
        if not self.activation_history:
            return 0.0
        
        # High variance + high average = important neuron
        activations = [h['value'] for h in self.activation_history]
        variance = np.var(activations)
        average = np.mean(activations)
        
        self.importance_score = (variance * 0.6 + average * 0.4)
        return self.importance_score

class ConnectionState:
    """Tracks the state of a connection between neurons"""
    def __init__(self, from_neuron: Tuple[int, int], to_neuron: Tuple[int, int]):
        self.from_neuron = from_neuron  # (layer, index)
        self.to_neuron = to_neuron
        self.weight_history = deque(maxlen=100)
        self.usage_count = 0
        self.gradient_flow = 0.0
    
    def update_weight(self, weight: float):
        """Update connection weight"""
        self.weight_history.append({
            'value': float(weight),
            'timestamp': datetime.now().isoformat()
        })
        self.usage_count += 1
    
    def get_current_weight(self) -> float:
        """Get most recent weight"""
        if self.weight_history:
            return self.weight_history[-1]['value']
        return 0.0
    
    def get_strength(self) -> float:
        """Calculate connection strength (abs weight * usage)"""
        if not self.weight_history:
            return 0.0
        return abs(self.get_current_weight()) * min(1.0, self.usage_count / 100)

class RealtimeVisualizer:
    """
    Real-time neural network visualizer
    Tracks and visualizes actual neuron activations, connection weights, and processing flow
    """
    def __init__(self):
        self.neurons = {}  # (layer, index) -> NeuronState
        self.connections = {}  # (from_neuron, to_neuron) -> ConnectionState
        self.processing_snapshots = deque(maxlen=50)
        self.genetic_patterns = []
        self.iteration_count = 0
        
    def initialize_from_network(self, network):
        """Initialize visualizer from neural network structure"""
        # Track neurons in each layer
        layer_sizes = [network.input_size] + network.hidden_sizes + [network.output_size]
        
        for layer_idx, size in enumerate(layer_sizes):
            for neuron_idx in range(size):
                neuron_id = (layer_idx, neuron_idx)
                self.neurons[neuron_id] = NeuronState(layer_idx, neuron_idx)
        
        # Initialize connections between layers
        for layer_idx in range(len(layer_sizes) - 1):
            for from_idx in range(layer_sizes[layer_idx]):
                for to_idx in range(layer_sizes[layer_idx + 1]):
                    from_neuron = (layer_idx, from_idx)
                    to_neuron = (layer_idx + 1, to_idx)
                    conn_id = (from_neuron, to_neuron)
                    self.connections[conn_id] = ConnectionState(from_neuron, to_neuron)
        
        print(f"[VISUALIZER] Initialized: {len(self.neurons)} neurons, {len(self.connections)} connections")
    
    def update_from_forward_pass(self, network, activations: List[np.ndarray]):
        """Update visualizer from network forward pass"""
        self.iteration_count += 1
        
        # Update neuron activations
        for layer_idx, layer_activations in enumerate(activations):
            if len(layer_activations.shape) == 2:  # Batch
                avg_activations = np.mean(layer_activations, axis=0)
            else:
                avg_activations = layer_activations
            
            for neuron_idx, activation in enumerate(avg_activations):
                if neuron_idx < len(avg_activations):
                    neuron_id = (layer_idx, neuron_idx)
                    if neuron_id in self.neurons:
                        self.neurons[neuron_id].update_activation(float(activation))
        
        # Update connection weights
        for layer_idx, weights in enumerate(network.weights):
            if layer_idx < len(network.weights):
                for from_idx in range(weights.shape[0]):
                    for to_idx in range(weights.shape[1]):
                        from_neuron = (layer_idx, from_idx)
                        to_neuron = (layer_idx + 1, to_idx)
                        conn_id = (from_neuron, to_neuron)
                        if conn_id in self.connections:
                            self.connections[conn_id].update_weight(float(weights[from_idx, to_idx]))
        
        # Calculate neuron importance
        for neuron in self.neurons.values():
            neuron.calculate_importance()
        
        # Create processing snapshot
        self._create_snapshot(network, activations)
    
    def _create_snapshot(self, network, activations: List[np.ndarray]):
        """Create a snapshot of current processing state"""
        snapshot = {
            'iteration': self.iteration_count,
            'timestamp': datetime.now().isoformat(),
            'active_neurons': [],
            'strong_connections': [],
            'layer_stats': []
        }
        
        # Identify highly active neurons
        for neuron_id, neuron in self.neurons.items():
            activation = neuron.get_current_activation()
            if activation > 0.5:  # Threshold for "active"
                snapshot['active_neurons'].append({
                    'layer': neuron_id[0],
                    'index': neuron_id[1],
                    'activation': float(activation),
                    'importance': float(neuron.importance_score)
                })
        
        # Identify strong connections being used
        for conn_id, conn in self.connections.items():
            strength = conn.get_strength()
            if strength > 0.3:  # Threshold for "strong"
                snapshot['strong_connections'].append({
                    'from': {'layer': conn_id[0][0], 'index': conn_id[0][1]},
                    'to': {'layer': conn_id[1][0], 'index': conn_id[1][1]},
                    'weight': float(conn.get_current_weight()),
                    'strength': float(strength)
                })
        
        # Layer-wise statistics
        layer_sizes = [network.input_size] + network.hidden_sizes + [network.output_size]
        for layer_idx, size in enumerate(layer_sizes):
            layer_neurons = [n for n_id, n in self.neurons.items() if n_id[0] == layer_idx]
            if layer_neurons:
                avg_activation = np.mean([n.get_current_activation() for n in layer_neurons])
                snapshot['layer_stats'].append({
                    'layer': layer_idx,
                    'size': size,
                    'avg_activation': float(avg_activation),
                    'active_count': len([n for n in layer_neurons if n.get_current_activation() > 0.5])
                })
        
        self.processing_snapshots.append(snapshot)
    
    def get_realtime_visualization_data(self) -> Dict[str, Any]:
        """Get current visualization data for dashboard"""
        # Get most recent snapshot
        current_snapshot = self.processing_snapshots[-1] if self.processing_snapshots else None
        
        # Get most important neurons
        important_neurons = sorted(
            [(n_id, n) for n_id, n in self.neurons.items()],
            key=lambda x: x[1].importance_score,
            reverse=True
        )[:20]
        
        # Get strongest connections
        strong_connections = sorted(
            [(c_id, c) for c_id, c in self.connections.items()],
            key=lambda x: x[1].get_strength(),
            reverse=True
        )[:30]
        
        return {
            'current_snapshot': current_snapshot,
            'important_neurons': [
                {
                    'layer': n_id[0],
                    'index': n_id[1],
                    'importance': float(n.importance_score),
                    'activation': float(n.get_current_activation()),
                    'avg_activation': float(n.get_avg_activation())
                }
                for n_id, n in important_neurons
            ],
            'strong_connections': [
                {
                    'from': {'layer': c_id[0][0], 'index': c_id[0][1]},
                    'to': {'layer': c_id[1][0], 'index': c_id[1][1]},
                    'weight': float(c.get_current_weight()),
                    'strength': float(c.get_strength()),
                    'usage': c.usage_count
                }
                for c_id, c in strong_connections
            ],
            'processing_flow': {
                'total_neurons': len(self.neurons),
                'active_neurons': len([n for n in self.neurons.values() if n.get_current_activation() > 0.5]),
                'total_connections': len(self.connections),
                'strong_connections': len([c for c in self.connections.values() if c.get_strength() > 0.3]),
                'iteration': self.iteration_count
            }
        }
    
    def track_genetic_pattern(self, pattern_type: str, description: str, fitness: float):
        """Track genetic learning patterns"""
        self.genetic_patterns.append({
            'type': pattern_type,
            'description': description,
            'fitness': fitness,
            'iteration': self.iteration_count,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_genetic_learning_data(self) -> Dict[str, Any]:
        """Get genetic learning visualization data"""
        return {
            'patterns': self.genetic_patterns[-20:],
            'total_patterns': len(self.genetic_patterns),
            'pattern_types': list(set([p['type'] for p in self.genetic_patterns])),
            'avg_fitness': np.mean([p['fitness'] for p in self.genetic_patterns]) if self.genetic_patterns else 0.0
        }
    
    def save_visualization_state(self, filepath: str = 'data/realtime_viz.json'):
        """Save current visualization state to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'realtime': self.get_realtime_visualization_data(),
            'genetic_learning': self.get_genetic_learning_data(),
            'snapshots': list(self.processing_snapshots),
            'last_updated': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def get_neuron_processing_path(self, input_data: np.ndarray, network) -> List[Dict]:
        """Trace the processing path through the network for visualization"""
        path = []
        
        # Forward pass through network
        activations = network.forward(input_data, training=False)
        
        # Track which neurons were most active at each layer
        for layer_idx in range(len(network.weights) + 1):
            if layer_idx < len(network.activations):
                layer_activations = network.activations[layer_idx]
                if len(layer_activations.shape) == 2:
                    avg_activations = np.mean(layer_activations, axis=0)
                else:
                    avg_activations = layer_activations
                
                # Get top active neurons in this layer
                top_indices = np.argsort(avg_activations)[-5:]
                
                path.append({
                    'layer': layer_idx,
                    'top_neurons': [
                        {
                            'index': int(idx),
                            'activation': float(avg_activations[idx])
                        }
                        for idx in top_indices
                    ]
                })
        
        return path
