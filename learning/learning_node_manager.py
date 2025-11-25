"""
LearningNodeManager: Manages learning nodes with specialization, pruning, and creation
Handles genetic learning with node-level pruning and creation
"""

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import os

@dataclass
class NodeData:
    id: str
    layer: int
    specialization: str  # expert, specialist, generalist, novice
    fitness: float
    age: int
    mutation_rate: float
    activation_function: str  # relu, sigmoid, tanh
    connections: int
    pruned: bool = False

    def to_dict(self):
        return asdict(self)

class LearningNodeManager:
    def __init__(self, initial_nodes: int = 20, max_nodes: int = 200):
        self.nodes: Dict[str, NodeData] = {}
        self.node_counter = 0
        self.max_nodes = max_nodes
        self.pruning_log = []
        self.creation_log = []
        self.specialization_map = {}

        # Initialize nodes
        for _ in range(initial_nodes):
            self._create_node()

    def _create_node(self) -> str:
        """Create a new learning node"""
        if len([n for n in self.nodes.values() if not n.pruned]) >= self.max_nodes:
            return None

        node_id = f"node_{self.node_counter}"
        self.node_counter += 1

        specialization = random.choice(['expert', 'specialist', 'generalist', 'novice'])
        activation = random.choice(['relu', 'sigmoid', 'tanh'])

        node = NodeData(
            id=node_id,
            layer=random.randint(0, 3),
            specialization=specialization,
            fitness=random.uniform(0.1, 0.9),
            age=0,
            mutation_rate=random.uniform(0.01, 0.2),
            activation_function=activation,
            connections=random.randint(1, 5)
        )

        self.nodes[node_id] = node
        self.creation_log.append({
            'node_id': node_id,
            'specialization': specialization,
            'timestamp': str(__import__('datetime').datetime.now())
        })

        return node_id

    def prune_node(self, node_id: str, reason: str = "Low fitness"):
        """Mark a node for pruning"""
        if node_id in self.nodes:
            self.nodes[node_id].pruned = True
            self.pruning_log.append({
                'node_id': node_id,
                'reason': reason,
                'timestamp': str(__import__('datetime').datetime.now())
            })

    def genetic_pruning(self, fitness_threshold: float = 0.3):
        """Prune nodes below fitness threshold"""
        pruned_count = 0
        for node_id, node in list(self.nodes.items()):
            if not node.pruned and node.fitness < fitness_threshold:
                self.prune_node(node_id, f"Fitness below threshold: {node.fitness:.2f}")
                pruned_count += 1

        return pruned_count

    def genetic_creation(self, population_size: int = None):
        """Create new nodes based on genetic success"""
        if population_size is None:
            current_active = len([n for n in self.nodes.values() if not n.pruned])
            population_size = max(20, int(current_active * 1.2))

        current_active = len([n for n in self.nodes.values() if not n.pruned])
        nodes_to_create = max(0, population_size - current_active)

        created = []
        for _ in range(nodes_to_create):
            node_id = self._create_node()
            if node_id:
                created.append(node_id)

        return created

    def update_node_fitness(self, node_id: str, fitness: float):
        """Update a node's fitness score"""
        if node_id in self.nodes:
            self.nodes[node_id].fitness = min(1.0, max(0.0, fitness))

    def age_nodes(self):
        """Increment age of all active nodes"""
        for node in self.nodes.values():
            if not node.pruned:
                node.age += 1

    def specialize_node(self, node_id: str, domain: str):
        """Specialize a node for a specific domain"""
        if node_id in self.nodes:
            if domain not in self.specialization_map:
                self.specialization_map[domain] = []
            self.specialization_map[domain].append(node_id)

    def get_expert_nodes(self, specialization: str = None) -> List[NodeData]:
        """Get expert nodes, optionally filtered by specialization"""
        experts = [n for n in self.nodes.values() if n.specialization == 'expert' and not n.pruned]
        if specialization:
            expert_ids = self.specialization_map.get(specialization, [])
            experts = [n for n in experts if n.id in expert_ids]
        return experts

    def get_node_connections(self, node_id: str) -> List[str]:
        """Get connected nodes (simulated)"""
        if node_id not in self.nodes:
            return []

        # Return random connections for simulation
        all_nodes = list(self.nodes.keys())
        all_nodes.remove(node_id)
        return random.sample(all_nodes, min(self.nodes[node_id].connections, len(all_nodes)))

    def get_network_topology(self) -> Dict:
        """Get network topology data"""
        active_nodes = [n for n in self.nodes.values() if not n.pruned]

        return {
            'active_nodes': len(active_nodes),
            'total_nodes': len(self.nodes),
            'pruned_nodes': len([n for n in self.nodes.values() if n.pruned]),
            'specialization_distribution': self._distribution_by_spec(),
            'layer_distribution': self._distribution_by_layer(),
            'average_fitness': sum(n.fitness for n in active_nodes) / len(active_nodes) if active_nodes else 0
        }

    def _distribution_by_spec(self) -> Dict[str, int]:
        """Get distribution of nodes by specialization"""
        dist = {}
        for node in self.nodes.values():
            if not node.pruned:
                dist[node.specialization] = dist.get(node.specialization, 0) + 1
        return dist

    def _distribution_by_layer(self) -> Dict[int, int]:
        """Get distribution of nodes by layer"""
        dist = {}
        for node in self.nodes.values():
            if not node.pruned:
                dist[node.layer] = dist.get(node.layer, 0) + 1
        return dist

    def export_nodes(self, filepath: str):
        """Export node data to JSON"""
        data = {
            'nodes': {nid: n.to_dict() for nid, n in self.nodes.items()},
            'topology': self.get_network_topology(),
            'pruning_log': self.pruning_log,
            'creation_log': self.creation_log
        }
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
