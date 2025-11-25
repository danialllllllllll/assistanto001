#!/usr/bin/env python3
"""
Replit Setup Script for Node Visualizer and Autonomous Code Evolution
Run this with: python setup_node_evolution.py
"""

import os
import json
from pathlib import Path

def create_file(filepath, content):
    """Create or update a file with the given content"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"✓ Created/Updated: {filepath}")

# ==============================================================================
# LEARNING MODULE: SELF EVOLVER
# ==============================================================================

self_evolver_content = '''"""
SelfEvolver: Autonomous code evolution system
Modifies AI code every 5 generations to optimize hyperparameters and architecture
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Tuple

class EvolutionSuggestion:
    def __init__(self, generation: int, change_type: str, target: str, old_value: Any, new_value: Any, reasoning: str):
        self.generation = generation
        self.change_type = change_type  # 'hyperparameter', 'architecture', 'learning_rate'
        self.target = target
        self.old_value = old_value
        self.new_value = new_value
        self.reasoning = reasoning
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            'generation': self.generation,
            'change_type': self.change_type,
            'target': self.target,
            'old_value': str(self.old_value),
            'new_value': str(self.new_value),
            'reasoning': self.reasoning,
            'timestamp': self.timestamp
        }

class CodeEvolutionLog:
    def __init__(self, log_file: str = 'learning/evolution_log.json'):
        self.log_file = log_file
        self.suggestions: List[EvolutionSuggestion] = []
        self.load()

    def load(self):
        """Load existing evolution log"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                data = json.load(f)
                self.suggestions = data.get('suggestions', [])

    def add_suggestion(self, suggestion: EvolutionSuggestion):
        """Record an evolution suggestion"""
        self.suggestions.append(suggestion)
        self.save()

    def save(self):
        """Persist evolution log to disk"""
        os.makedirs(os.path.dirname(self.log_file) or '.', exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump({
                'suggestions': [s.to_dict() if isinstance(s, EvolutionSuggestion) else s for s in self.suggestions],
                'total_changes': len(self.suggestions)
            }, f, indent=2)

    def get_changes_by_generation(self, generation: int) -> List[Dict]:
        """Get all changes made at a specific generation"""
        return [s.to_dict() if isinstance(s, EvolutionSuggestion) else s for s in self.suggestions if (s.generation if isinstance(s, EvolutionSuggestion) else s.get('generation')) == generation]

class SelfEvolver:
    def __init__(self, log_file: str = 'learning/evolution_log.json'):
        self.evolution_log = CodeEvolutionLog(log_file)
        self.hyperparameters = {
            'learning_rate': 0.001,
            'mutation_rate': 0.1,
            'population_size': 50,
            'tournament_size': 3,
            'elite_size': 2
        }
        self.generation_counter = 0
        self.fitness_history = []

    def should_evolve(self, generation: int) -> bool:
        """Check if it's time to evolve (every 5 generations)"""
        return generation > 0 and generation % 5 == 0

    def generate_evolution_suggestions(self, current_fitness: float, prev_fitness: float, generation: int) -> List[EvolutionSuggestion]:
        """Generate code evolution suggestions based on fitness"""
        suggestions = []
        fitness_improvement = current_fitness - prev_fitness

        # Suggest learning rate adjustment
        if fitness_improvement < 0.01:  # Stagnation
            new_lr = self.hyperparameters['learning_rate'] * 1.5
            suggestions.append(EvolutionSuggestion(
                generation=generation,
                change_type='hyperparameter',
                target='learning_rate',
                old_value=self.hyperparameters['learning_rate'],
                new_value=new_lr,
                reasoning='Fitness stagnation detected. Increasing learning rate to escape local optimum.'
            ))
            self.hyperparameters['learning_rate'] = new_lr
        elif fitness_improvement > 0.1:  # Good progress
            new_lr = self.hyperparameters['learning_rate'] * 0.95
            suggestions.append(EvolutionSuggestion(
                generation=generation,
                change_type='hyperparameter',
                target='learning_rate',
                old_value=self.hyperparameters['learning_rate'],
                new_value=new_lr,
                reasoning='Strong fitness improvement. Fine-tuning learning rate for precision.'
            ))
            self.hyperparameters['learning_rate'] = new_lr

        # Suggest mutation rate adjustment
        if current_fitness > 0.9:
            new_mutation = self.hyperparameters['mutation_rate'] * 0.8
            suggestions.append(EvolutionSuggestion(
                generation=generation,
                change_type='hyperparameter',
                target='mutation_rate',
                old_value=self.hyperparameters['mutation_rate'],
                new_value=new_mutation,
                reasoning='High fitness achieved. Reducing mutation to refine solutions.'
            ))
            self.hyperparameters['mutation_rate'] = new_mutation
        elif current_fitness < 0.5:
            new_mutation = self.hyperparameters['mutation_rate'] * 1.2
            suggestions.append(EvolutionSuggestion(
                generation=generation,
                change_type='hyperparameter',
                target='mutation_rate',
                old_value=self.hyperparameters['mutation_rate'],
                new_value=new_mutation,
                reasoning='Low fitness detected. Increasing mutation for exploration.'
            ))
            self.hyperparameters['mutation_rate'] = new_mutation

        # Log all suggestions
        for suggestion in suggestions:
            self.evolution_log.add_suggestion(suggestion)

        return suggestions

    def evolve_architecture(self, current_nodes: int, generation: int) -> Dict:
        """Suggest architecture changes"""
        suggestion = None

        if current_nodes < 10:
            new_nodes = int(current_nodes * 1.5)
            suggestion = EvolutionSuggestion(
                generation=generation,
                change_type='architecture',
                target='num_nodes',
                old_value=current_nodes,
                new_value=new_nodes,
                reasoning='Insufficient nodes. Expanding network capacity for better learning.'
            )
        elif current_nodes > 100:
            new_nodes = int(current_nodes * 0.8)
            suggestion = EvolutionSuggestion(
                generation=generation,
                change_type='architecture',
                target='num_nodes',
                old_value=current_nodes,
                new_value=new_nodes,
                reasoning='Excessive nodes causing overfitting. Pruning network.'
            )

        if suggestion:
            self.evolution_log.add_suggestion(suggestion)
            return suggestion.to_dict()

        return {}

    def get_evolution_state(self) -> Dict:
        """Get current evolution state"""
        return {
            'hyperparameters': self.hyperparameters,
            'generation': self.generation_counter,
            'fitness_history': self.fitness_history,
            'total_changes': len(self.evolution_log.suggestions),
            'evolution_log': [s.to_dict() if isinstance(s, EvolutionSuggestion) else s for s in self.evolution_log.suggestions]
        }
'''

# ==============================================================================
# LEARNING MODULE: NODE VISUALIZER
# ==============================================================================

node_visualizer_content = '''"""
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
'''

# ==============================================================================
# LEARNING MODULE: LEARNING NODE MANAGER
# ==============================================================================

learning_node_manager_content = '''"""
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
'''

# ==============================================================================
# Create all files
# ==============================================================================

print("Setting up Node Visualizer and Autonomous Evolution System...\n")

create_file('learning/self_evolver.py', self_evolver_content)
create_file('learning/node_visualizer.py', node_visualizer_content)
create_file('learning/learning_node_manager.py', learning_node_manager_content)

# Create __init__.py if it doesn't exist
init_content = '''"""Learning module for AI evolution and training"""

from .self_evolver import SelfEvolver, CodeEvolutionLog, EvolutionSuggestion
from .node_visualizer import NodeVisualizer, NodeVisualizerData
from .learning_node_manager import LearningNodeManager, NodeData

__all__ = [
    'SelfEvolver',
    'CodeEvolutionLog',
    'EvolutionSuggestion',
    'NodeVisualizer',
    'NodeVisualizerData',
    'LearningNodeManager',
    'NodeData'
]
'''

create_file('learning/__init__.py', init_content)

print("\n" + "="*70)
print("✓ Setup Complete!")
print("="*70)
print("\nNew files created:")
print("  • learning/self_evolver.py - Autonomous code evolution")
print("  • learning/node_visualizer.py - Network topology visualization")
print("  • learning/learning_node_manager.py - Learning node management")
print("  • learning/__init__.py - Module initialization")
print("\nTo use in your train_advanced_ai.py:")
print("  from learning import SelfEvolver, NodeVisualizer, LearningNodeManager")
print("\nExample integration:")
print("  evolver = SelfEvolver()")
print("  visualizer = NodeVisualizer()")
print("  node_manager = LearningNodeManager(initial_nodes=20)")
print("\nEvery 5 generations, the AI will autonomously evolve its hyperparameters!")
print("Node visualization data is exported to: data/")
'''

# ==============================================================================
# Create installation instruction file
# ==============================================================================

install_instructions = """# Node Visualizer & Autonomous Evolution Setup

## Quick Start

Run this command in your Replit shell:

```bash
python setup_node_evolution.py
'''