import networkx as nx
import json
import numpy as np
from typing import Dict, List

class LearningNodeVisualizer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_stats = {}

    def update_node(self, node_id: str, data: Dict):
        """Update node with learning metrics."""
        self.node_stats[node_id] = {
            'understanding': data.get('understanding', 0),
            'confidence': data.get('confidence', 0),
            'connections': data.get('connections', []),
            'weight': self.calculate_node_weight(data)
        }
        self.graph.add_node(node_id)
        for conn in data.get('connections', []):
            self.graph.add_edge(node_id, conn)

    def calculate_node_weight(self, data: Dict) -> float:
        """Calculate node weight based on metrics."""
        understanding = data.get('understanding', 0)
        confidence = data.get('confidence', 0)
        accuracy = data.get('accuracy', 0)
        return max(0.1, min(1.0, (understanding * 0.4 + confidence * 0.3 + accuracy * 0.3)))

    def generate_graph_data(self) -> Dict:
        """Generate graph data for visualization."""
        nodes = []
        edges = []
        pos = nx.spring_layout(self.graph, k=2)
        for node_id, stats in self.node_stats.items():
            nodes.append({
                'id': node_id,
                'size': stats['weight'] * 30,
                'x': float(pos[node_id][0] * 500),
                'y': float(pos[node_id][1] * 500),
                'color': '#4CAF50' if stats['understanding'] >= 0.9 else '#FFC107' if stats['understanding'] >= 0.6 else '#F44336'
            })
            for conn in stats['connections']:
                if conn in self.node_stats:
                    edges.append({
                        'source': node_id,
                        'target': conn,
                        'weight': min(stats['weight'], self.node_stats[conn]['weight'])
                    })
        return {'nodes': nodes, 'edges': edges}