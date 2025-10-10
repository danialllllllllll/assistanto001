import networkx as nx
import json
import numpy as np
from typing import Dict, List

class LearningNodeVisualizer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_stats = {}

    def update_node(self, node_id: str, data: Dict):
        """Update node with real learning metrics"""
        self.node_stats[node_id] = {
            'understanding': data.get('understanding', 0),
            'confidence': data.get('confidence', 0),
            'connections': data.get('connections', []),
            'weight': self.calculate_node_weight(data)
        }

    def calculate_node_weight(self, data: Dict) -> float:
        """Calculate meaningful node weight based on learning metrics"""
        understanding = data.get('understanding', 0)
        confidence = data.get('confidence', 0)
        accuracy = data.get('accuracy', 0)

        weight = (understanding * 0.4 + 
                 confidence * 0.3 + 
                 accuracy * 0.3)

        return max(0.1, min(1.0, weight))

    def generate_graph_data(self) -> Dict:
        """Generate graph data with meaningful node sizes and positions"""
        nodes = []
        edges = []

        pos = nx.spring_layout(self.graph, k=2)

        for node_id, stats in self.node_stats.items():
            nodes.append({
                'id': node_id,
                'size': stats['weight'] * 30,
                'x': float(pos[node_id][0] * 500),
                'y': float(pos[node_id][1] * 500),
                'color': self.get_node_color(stats)
            })

            for conn in stats['connections']:
                if conn in self.node_stats:
                    edges.append({
                        'source': node_id,
                        'target': conn,
                        'weight': min(stats['weight'], 
                                    self.node_stats[conn]['weight'])
                    })

        return {
            'nodes': nodes,
            'edges': edges
        }

    def get_node_color(self, stats: Dict) -> str:
        """Color nodes based on learning progress"""
        understanding = stats['understanding']
        if understanding >= 0.9:
            return '#4CAF50'  # Green for high understanding
        elif understanding >= 0.6:
            return '#FFC107'  # Yellow for medium understanding
        return '#F44336'  # Red for low understanding