"""
Node Visualizer: Generates visualization data for learning node network
Provides topology, evolution timeline, genetic landscape, and node statistics
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime

class NodeVisualizerData:
    def __init__(self):
        self.nodes_data = {}
        self.connections = []
        self.evolution_timeline = []
        self.pruning_history = []

    def add_node(self, node_id: str, layer: int, specialization: str, fitness: float, age: int):
        """Add node to visualization"""
        self.nodes_data[node_id] = {
            'id': node_id,
            'layer': layer,
            'specialization': specialization,
            'fitness': fitness,
            'age': age,
            'color': self._get_color_by_specialization(specialization),
            'timestamp': datetime.now().isoformat()
        }

    def add_connection(self, from_id: str, to_id: str, weight: float):
        """Add connection between nodes"""
        self.connections.append({
            'source': from_id,
            'target': to_id,
            'weight': weight,
            'thickness': min(3, abs(weight) * 2) + 0.5
        })

    def add_pruning(self, node_id: str, reason: str):
        """Record node pruning"""
        self.pruning_history.append({
            'node_id': node_id,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })

    def add_evolution_step(self, generation: int, improvement: float, changes: List[str]):
        """Record evolution milestone"""
        self.evolution_timeline.append({
            'generation': generation,
            'improvement': improvement,
            'changes': changes,
            'timestamp': datetime.now().isoformat()
        })

    def _get_color_by_specialization(self, specialization: str) -> str:
        """Map specialization to color"""
        colors = {
            'expert': '#FF6B6B',      # Red
            'specialist': '#4ECDC4',  # Teal
            'generalist': '#95E1D3',  # Light Teal
            'novice': '#FFE66D',      # Yellow
            'default': '#A8A8A8'      # Gray
        }
        return colors.get(specialization, colors['default'])

    def get_topology_visualization(self) -> Dict:
        """Get network topology for visualization"""
        return {
            'nodes': list(self.nodes_data.values()),
            'connections': self.connections,
            'stats': {
                'total_nodes': len(self.nodes_data),
                'total_connections': len(self.connections),
                'layers': max([n['layer'] for n in self.nodes_data.values()], default=0) + 1 if self.nodes_data else 0
            }
        }

    def get_evolution_timeline(self) -> List[Dict]:
        """Get evolution timeline for visualization"""
        return sorted(self.evolution_timeline, key=lambda x: x['generation'])

    def get_pruning_history(self) -> List[Dict]:
        """Get pruning history"""
        return self.pruning_history

    def get_node_statistics(self, node_id: str) -> Optional[Dict]:
        """Get detailed statistics for a specific node"""
        return self.nodes_data.get(node_id)

    def get_genetic_landscape(self) -> Dict:
        """Get genetic algorithm landscape data"""
        fitness_values = [n['fitness'] for n in self.nodes_data.values()]
        if not fitness_values:
            return {'average_fitness': 0, 'max_fitness': 0, 'min_fitness': 0}

        return {
            'average_fitness': sum(fitness_values) / len(fitness_values),
            'max_fitness': max(fitness_values),
            'min_fitness': min(fitness_values),
            'nodes_by_specialization': self._nodes_by_specialization(),
            'nodes_by_layer': self._nodes_by_layer()
        }

    def _nodes_by_specialization(self) -> Dict[str, int]:
        """Count nodes by specialization"""
        counts = {}
        for node in self.nodes_data.values():
            spec = node['specialization']
            counts[spec] = counts.get(spec, 0) + 1
        return counts

    def _nodes_by_layer(self) -> Dict[int, int]:
        """Count nodes by layer"""
        counts = {}
        for node in self.nodes_data.values():
            layer = node['layer']
            counts[layer] = counts.get(layer, 0) + 1
        return counts

    def export_json(self, filepath: str):
        """Export all visualization data to JSON"""
        data = {
            'topology': self.get_topology_visualization(),
            'evolution_timeline': self.get_evolution_timeline(),
            'pruning_history': self.get_pruning_history(),
            'genetic_landscape': self.get_genetic_landscape()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

class NodeVisualizer:
    def __init__(self):
        self.viz_data = NodeVisualizerData()

    def create_network_topology_data(self, nodes: List[Dict], connections: List[Dict]) -> Dict:
        """Create network topology from node and connection data"""
        # Clear existing
        self.viz_data.nodes_data = {}
        self.viz_data.connections = []

        # Add nodes
        for node in nodes:
            self.viz_data.add_node(
                node['id'],
                node.get('layer', 0),
                node.get('specialization', 'default'),
                node.get('fitness', 0.5),
                node.get('age', 0)
            )

        # Add connections
        for conn in connections:
            self.viz_data.add_connection(
                conn['source'],
                conn['target'],
                conn.get('weight', 1.0)
            )

        return self.viz_data.get_topology_visualization()

    def export_for_web(self, output_dir: str = 'data'):
        """Export visualization data for web dashboard"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Export topology
        with open(f'{output_dir}/node_topology.json', 'w') as f:
            json.dump(self.viz_data.get_topology_visualization(), f)

        # Export evolution
        with open(f'{output_dir}/evolution_timeline.json', 'w') as f:
            json.dump(self.viz_data.get_evolution_timeline(), f)

        # Export genetic landscape
        with open(f'{output_dir}/genetic_landscape.json', 'w') as f:
            json.dump(self.viz_data.get_genetic_landscape(), f)

        return {
            'topology_file': f'{output_dir}/node_topology.json',
            'evolution_file': f'{output_dir}/evolution_timeline.json',
            'landscape_file': f'{output_dir}/genetic_landscape.json'
        }
