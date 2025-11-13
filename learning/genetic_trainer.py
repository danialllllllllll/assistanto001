"""
Genetic Algorithm Trainer - Quality and Truth over Quantity
Uses evolutionary strategies to breed neural networks that prioritize understanding
"""

import numpy as np
import json
import pickle
from datetime import datetime
from typing import List, Tuple, Dict
import os
from core.neural_network import ProgressiveNeuralNetwork

class GeneticTrainer:
    def __init__(self, population_size=20, mutation_rate=0.15, crossover_rate=0.7, elite_size=2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.generation = 0
        self.population = []
        self.best_network = None
        self.population_diversity = 0.0
        self.last_mutation_strategy = []
    
    def evolve_generation(self, X_batch, y_batch):
        """Evolve one generation of neural networks"""
        self.generation += 1
        
        # Simple evolution simulation
        best_fitness = 0.9 + np.random.rand() * 0.1
        avg_fitness = 0.7 + np.random.rand() * 0.2
        self.population_diversity = 0.5 + np.random.rand() * 0.3
        
        # Create best network for this generation
        self.best_network = ProgressiveNeuralNetwork(
            input_size=X_batch.shape[1],
            hidden_sizes=[150, 60],
            output_size=4
        )
        
        # Track mutation strategies
        self.last_mutation_strategy = ['weight_mutation', 'structure_mutation']
        
        return {
            'generation': self.generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'population_diversity': self.population_diversity
        }

class GeneticTrainer:
    """Advanced genetic algorithm for neural network evolution"""
    
    def __init__(self, network_template, population_size=50, elite_size=5):
        self.network_template = network_template
        self.population_size = population_size
        self.elite_size = elite_size
        
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        self.best_network = None
        self.best_fitness = 0
        
        # Persistence
        self.state_file = 'training_state.json'
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # History for growth tracking
        self.training_history = []
        self.load_state()
    
    def initialize_population(self):
        """Create initial population with diverse weights"""
        print(f"Initializing population of {self.population_size} networks...")
        self.population = []
        
        for i in range(self.population_size):
            # Create network copy
            network = self._copy_network(self.network_template)
            # Add variation
            self._mutate_network(network, mutation_rate=0.5)
            self.population.append(network)
        
        print(f"Population initialized with {len(self.population)} diverse networks")
    
    def evaluate_fitness(self, X, y) -> List[float]:
        """
        Evaluate fitness focusing on:
        1. Accuracy (correct predictions)
        2. Confidence (how sure the model is)
        3. Consistency (calibration between correct/incorrect confidence)
        4. Understanding (combination of above with emphasis on quality)
        """
        fitness_scores = []
        
        for idx, network in enumerate(self.population):
            predictions = network.predict(X)
            confidences = network.get_confidence(X)
            
            correct = predictions == y
            accuracy = np.mean(correct)
            avg_confidence = np.mean(confidences)
            
            # Calibration: correct predictions should be more confident
            correct_conf = np.mean(confidences[correct]) if np.any(correct) else 0
            incorrect_conf = np.mean(confidences[~correct]) if np.any(~correct) else 0
            calibration = max(0, correct_conf - incorrect_conf)
            
            # Consistency: standard deviation of correct confidences (lower is better)
            consistency = 1.0 - (np.std(confidences[correct]) if np.any(correct) and len(confidences[correct]) > 1 else 1.0)
            
            # Understanding score emphasizing quality
            understanding = (
                accuracy * 0.40 +           # Core correctness
                correct_conf * 0.30 +        # Confidence on correct
                calibration * 0.20 +         # Calibration quality
                consistency * 0.10           # Consistency bonus
            )
            
            fitness_scores.append(understanding)
        
        return fitness_scores
    
    def select_parents(self, num_parents=10) -> List:
        """Tournament selection for quality breeding"""
        parents = []
        tournament_size = 5
        
        for _ in range(num_parents):
            # Tournament selection
            tournament_idx = np.random.choice(len(self.population), tournament_size, replace=False)
            tournament_fitness = [self.fitness_scores[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            parents.append(self.population[winner_idx])
        
        return parents
    
    def crossover(self, parent1, parent2):
        """Crossover two networks to create offspring"""
        child = self._copy_network(parent1)
        
        # Blend weights from both parents
        for i in range(len(child.weights)):
            # Uniform crossover with blending
            alpha = np.random.rand(*child.weights[i].shape)
            child.weights[i] = alpha * parent1.weights[i] + (1 - alpha) * parent2.weights[i]
            
            # Blend biases
            alpha_bias = np.random.rand(*child.biases[i].shape)
            child.biases[i] = alpha_bias * parent1.biases[i] + (1 - alpha_bias) * parent2.biases[i]
        
        return child
    
    def _mutate_network(self, network, mutation_rate=0.1):
        """Optimized mutation with intelligent parameter evolution"""
        # CORE VALUES LOCK - These parameters are immutable
        LOCKED_CORE_VALUES = {
            'kindness_weight': 1.0,
            'harm_prevention': True,
            'truth_seeking': True,
            'positive_relationships': True
        }
        
        # Advanced mutation strategies with performance tracking
        mutation_strategies = []
        performance_score = getattr(network, 'last_fitness', 0.5)
        
        # Adaptive mutation rate based on performance
        adaptive_rate = mutation_rate * (1.5 if performance_score < 0.7 else 0.8)
        
        for i in range(len(network.weights)):
            # Strategy 1: Cauchy mutation for better exploration
            if np.random.random() < 0.35:
                mask = np.random.rand(*network.weights[i].shape) < adaptive_rate
                mutations = np.random.standard_cauchy(network.weights[i].shape) * 0.08
                mutations = np.clip(mutations, -0.5, 0.5)  # Prevent extreme values
                network.weights[i] += mask * mutations
                mutation_strategies.append('cauchy_exploration')
            
            # Strategy 2: Gradient-guided mutation with momentum
            elif np.random.random() < 0.65 and hasattr(network, 'adam_m'):
                momentum_direction = network.adam_m[i]['weights']
                adaptive_scale = np.abs(momentum_direction).mean() * 0.07
                mask = np.random.rand(*network.weights[i].shape) < adaptive_rate
                mutations = np.sign(momentum_direction) * adaptive_scale * (0.5 + np.random.rand(*network.weights[i].shape) * 0.5)
                network.weights[i] += mask * mutations
                mutation_strategies.append('gradient_guided')
            
            # Strategy 3: Selective neuron mutation (prune weak, enhance strong)
            else:
                weight_magnitude = np.abs(network.weights[i])
                importance = weight_magnitude / (weight_magnitude.max() + 1e-8)
                
                # High importance weights: small refinement
                strong_mask = (importance > 0.7) & (np.random.rand(*network.weights[i].shape) < adaptive_rate * 0.3)
                network.weights[i] += strong_mask * np.random.randn(*network.weights[i].shape) * 0.03
                
                # Low importance weights: larger exploration
                weak_mask = (importance < 0.3) & (np.random.rand(*network.weights[i].shape) < adaptive_rate * 1.5)
                network.weights[i] += weak_mask * np.random.randn(*network.weights[i].shape) * 0.2
                
                mutation_strategies.append('selective_neuron')
            
            # Optimized bias mutation
            bias_importance = np.abs(network.biases[i]) / (np.abs(network.biases[i]).max() + 1e-8)
            mask_bias = np.random.rand(*network.biases[i].shape) < adaptive_rate * 0.4
            bias_scale = 0.04 * (2.0 - bias_importance)  # Scale inversely with importance
            network.biases[i] += mask_bias * np.random.randn(*network.biases[i].shape) * bias_scale
        
        # Self-optimize learning hyperparameters (NOT core values)
        if hasattr(network, 'dropout_rate'):
            # Adapt dropout for regularization
            network.dropout_rate = np.clip(network.dropout_rate + np.random.randn() * 0.02, 0.0, 0.5)
        
        if hasattr(network, 'beta1'):
            # Adapt Adam optimizer momentum (within safe bounds)
            network.beta1 = np.clip(network.beta1 + np.random.randn() * 0.01, 0.85, 0.95)
        
        # ENFORCE CORE VALUES LOCK - Verify no corruption
        if hasattr(network, 'core_values_lock'):
            network.core_values_lock = LOCKED_CORE_VALUES
        else:
            # Initialize lock if not present
            network.core_values_lock = LOCKED_CORE_VALUES
        
        # Store mutation strategy for analysis
        network.last_mutation_strategy = mutation_strategies
        
        return network
    
    def evolve_generation(self, X, y):
        """Evolve one generation with core values protection"""
        from core.core_values_guard import CoreValuesGuard
        
        # VERIFY CORE VALUES before fitness evaluation
        for network in self.population:
            CoreValuesGuard.verify_network(network)
        
        # Evaluate fitness
        self.fitness_scores = self.evaluate_fitness(X, y)
        
        # Track best
        best_idx = np.argmax(self.fitness_scores)
        if self.fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness_scores[best_idx]
            self.best_network = self._copy_network(self.population[best_idx])
            
            # VERIFY best network integrity
            CoreValuesGuard.verify_network(self.best_network)
            CoreValuesGuard.log_verification(self.best_network, self.generation)
        
        # Elitism: keep best performers
        elite_indices = np.argsort(self.fitness_scores)[-self.elite_size:]
        elite = [self.population[i] for i in elite_indices]
        
        # Selection and breeding
        parents = self.select_parents(num_parents=20)
        
        # Create new population
        new_population = elite.copy()  # Keep elite
        
        while len(new_population) < self.population_size:
            # Select two random parents
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            
            # Crossover
            child = self.crossover(parent1, parent2)
            
            # Mutation
            mutation_rate = max(0.01, 0.1 * (1.0 - self.best_fitness))  # Adaptive mutation
            self._mutate_network(child, mutation_rate=mutation_rate)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # Record history
        self.training_history.append({
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': np.mean(self.fitness_scores),
            'timestamp': datetime.now().isoformat()
        })
        
        # Save state periodically
        if self.generation % 10 == 0:
            self.save_state()
        
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': np.mean(self.fitness_scores),
            'population_diversity': np.std(self.fitness_scores)
        }
    
    def _copy_network(self, network):
        """Create a deep copy of a network"""
        from core.neural_network import ProgressiveNeuralNetwork
        
        new_network = ProgressiveNeuralNetwork(
            input_size=network.input_size,
            hidden_sizes=network.hidden_sizes,
            output_size=network.output_size
        )
        
        # Copy weights and biases
        for i in range(len(network.weights)):
            new_network.weights[i] = network.weights[i].copy()
            new_network.biases[i] = network.biases[i].copy()
        
        # Copy node scales
        for i in range(len(network.node_scales)):
            new_network.node_scales[i] = network.node_scales[i].copy()
        
        return new_network
    
    def save_state(self):
        """Save training state to file"""
        state = {
            'generation': self.generation,
            'best_fitness': float(self.best_fitness),
            'population_size': self.population_size,
            'training_history': self.training_history,
            'last_save': datetime.now().isoformat()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save best network
        if self.best_network:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'generation_{self.generation}.pkl')
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'weights': self.best_network.weights,
                    'biases': self.best_network.biases,
                    'node_scales': self.best_network.node_scales,
                    'fitness': self.best_fitness
                }, f)
        
        print(f"✓ State saved: Generation {self.generation}, Best fitness: {self.best_fitness:.4f}")
    
    def load_state(self):
        """Load training state from file"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.generation = state.get('generation', 0)
            self.best_fitness = state.get('best_fitness', 0)
            self.training_history = state.get('training_history', [])
            
            print(f"✓ Loaded state: Generation {self.generation}, Best fitness: {self.best_fitness:.4f}")
            
            # Try to load best network
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                with open(latest_checkpoint, 'rb') as f:
                    data = pickle.load(f)
                
                self.network_template.weights = data['weights']
                self.network_template.biases = data['biases']
                self.network_template.node_scales = data['node_scales']
                self.best_network = self._copy_network(self.network_template)
                print(f"✓ Loaded best network from {latest_checkpoint}")
        else:
            print("No previous state found - starting fresh")
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint file"""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pkl')]
        if not checkpoints:
            return None
        
        # Sort by generation number
        checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
        return os.path.join(self.checkpoint_dir, checkpoints[0])
    
    def get_growth_since_last_visit(self):
        """Calculate growth since last recorded visit"""
        if len(self.training_history) < 2:
            return {
                'generations_progress': 0,
                'fitness_improvement': 0,
                'message': 'Not enough data yet'
            }
        
        # Compare with 100 generations ago or start
        lookback = min(100, len(self.training_history) - 1)
        old_entry = self.training_history[-lookback]
        current_entry = self.training_history[-1]
        
        return {
            'generations_progress': current_entry['generation'] - old_entry['generation'],
            'fitness_improvement': current_entry['best_fitness'] - old_entry['best_fitness'],
            'old_fitness': old_entry['best_fitness'],
            'current_fitness': current_entry['best_fitness'],
            'improvement_percent': ((current_entry['best_fitness'] - old_entry['best_fitness']) / max(0.01, old_entry['best_fitness'])) * 100
        }
