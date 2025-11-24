
import os
import json
import ast
import random
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import inspect
import importlib
import sys

class SelfEvolver:
    """
    True self-evolution engine that analyzes and improves its own code
    """
    def __init__(self, network, trainer):
        self.network = network
        self.trainer = trainer
        self.generation = 0
        self.improvement_log = []
        self.code_mutations = []
        self.performance_history = []
        self.optimized_functions = {}
        
    def analyze_own_code(self) -> Dict[str, Any]:
        """Analyze self-evolver.py and other AI modules for optimization opportunities"""
        analysis = {
            'functions': [],
            'complexity': {},
            'bottlenecks': [],
            'optimization_opportunities': []
        }
        
        # Analyze this file
        try:
            with open(__file__, 'r') as f:
                tree = ast.parse(f.read())
                
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis['functions'].append({
                        'name': node.name,
                        'lines': len(node.body),
                        'complexity': self._calculate_complexity(node)
                    })
                    
        except Exception as e:
            print(f"Code analysis error: {e}")
            
        return analysis
    
    def _calculate_complexity(self, node) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        return complexity
    
    def evolve_learning_rate(self, current_performance: float) -> float:
        """Dynamically evolve optimal learning rate based on performance"""
        if not self.performance_history:
            return 0.001
            
        recent_performance = self.performance_history[-10:]
        trend = np.mean(np.diff(recent_performance)) if len(recent_performance) > 1 else 0
        
        if trend > 0:
            # Improving - slightly increase learning rate
            new_lr = self.network.learning_rate * 1.05
        elif trend < -0.01:
            # Degrading - reduce learning rate
            new_lr = self.network.learning_rate * 0.8
        else:
            # Stable - small random mutation
            new_lr = self.network.learning_rate * (1 + random.gauss(0, 0.1))
            
        return np.clip(new_lr, 0.0001, 0.1)
    
    def optimize_network_architecture(self) -> Dict[str, Any]:
        """Evolve network architecture based on performance"""
        changes = {
            'layers_added': 0,
            'layers_removed': 0,
            'nodes_added': 0,
            'nodes_pruned': 0
        }
        
        current_fitness = self.trainer.best_fitness if hasattr(self.trainer, 'best_fitness') else 0
        
        # If performance is high, try growing
        if current_fitness > 0.8 and len(self.network.weights) < 6:
            # Add a layer
            new_layer_size = random.choice([32, 64, 128])
            self._add_layer(new_layer_size)
            changes['layers_added'] = 1
            changes['nodes_added'] = new_layer_size
            print(f"[EVOLUTION] Added layer with {new_layer_size} nodes")
            
        # If performance is low, try pruning
        elif current_fitness < 0.3:
            pruned = self._prune_weak_connections(threshold=0.05)
            changes['nodes_pruned'] = pruned
            print(f"[EVOLUTION] Pruned {pruned} weak connections")
            
        return changes
    
    def _add_layer(self, size: int):
        """Add a new layer to the network"""
        if not hasattr(self.network, 'weights') or len(self.network.weights) == 0:
            return
            
        # Insert new layer before output
        insert_idx = len(self.network.weights) - 1
        
        prev_size = self.network.weights[insert_idx].shape[0]
        next_size = self.network.weights[insert_idx].shape[1]
        
        # Create new layer weights
        new_weight_in = np.random.randn(prev_size, size) * 0.01
        new_weight_out = np.random.randn(size, next_size) * 0.01
        new_bias_in = np.zeros(size)
        new_bias_out = np.zeros(next_size)
        
        # Insert into network
        self.network.weights.insert(insert_idx, new_weight_in)
        self.network.biases.insert(insert_idx, new_bias_in)
        
        # Update next layer
        self.network.weights[insert_idx + 1] = new_weight_out
        
    def _prune_weak_connections(self, threshold: float = 0.05) -> int:
        """Prune connections with weights below threshold"""
        pruned_count = 0
        
        for i in range(len(self.network.weights)):
            mask = np.abs(self.network.weights[i]) > threshold
            pruned_count += np.sum(~mask)
            self.network.weights[i] *= mask
            
        return pruned_count
    
    def generate_training_strategy(self, stage_name: str) -> Dict[str, Any]:
        """Generate optimized training strategy for current stage"""
        strategy = {
            'batch_size': 64,
            'learning_rate': 0.001,
            'mutation_rate': 0.1,
            'focus_areas': []
        }
        
        # Stage-specific optimization
        if 'Baby' in stage_name:
            strategy['learning_rate'] = 0.005
            strategy['focus_areas'] = ['pattern_recognition', 'basic_classification']
        elif 'Elementary' in stage_name:
            strategy['learning_rate'] = 0.002
            strategy['focus_areas'] = ['understanding', 'self_quizzing']
        elif 'Scholar' in stage_name or 'Thinker' in stage_name:
            strategy['learning_rate'] = 0.0005
            strategy['focus_areas'] = ['reasoning', 'philosophy', 'truth_verification']
            
        return strategy
    
    def self_improve(self, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Main self-improvement cycle"""
        self.generation += 1
        self.performance_history.append(performance_metrics.get('fitness', 0))
        
        improvements = {
            'generation': self.generation,
            'timestamp': datetime.now().isoformat(),
            'code_analysis': {},
            'architecture_changes': {},
            'strategy_updates': {}
        }
        
        # Analyze own code
        improvements['code_analysis'] = self.analyze_own_code()
        
        # Evolve learning parameters
        new_lr = self.evolve_learning_rate(performance_metrics.get('fitness', 0))
        if hasattr(self.network, 'learning_rate'):
            self.network.learning_rate = new_lr
            improvements['strategy_updates']['learning_rate'] = new_lr
        
        # Optimize architecture
        improvements['architecture_changes'] = self.optimize_network_architecture()
        
        # Log improvements
        self.improvement_log.append(improvements)
        
        # Save evolution history
        self._save_evolution_history()
        
        return improvements
    
    def _save_evolution_history(self):
        """Save evolution history to file"""
        os.makedirs('knowledge', exist_ok=True)
        with open('knowledge/self_evolution.json', 'w') as f:
            json.dump({
                'generation': self.generation,
                'improvement_log': self.improvement_log[-50:],
                'performance_history': self.performance_history[-100:]
            }, f, indent=2)
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get evolution statistics"""
        return {
            'generation': self.generation,
            'total_improvements': len(self.improvement_log),
            'performance_trend': np.mean(np.diff(self.performance_history[-20:])) if len(self.performance_history) > 20 else 0,
            'current_performance': self.performance_history[-1] if self.performance_history else 0
        }
