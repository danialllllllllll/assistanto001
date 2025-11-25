"""
Genetic Learning Algorithm Integration
Implements human-like adaptive learning using genetic algorithms
Evolves neural network patterns similar to biological learning
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple
from datetime import datetime
from collections import deque

class LearningGenome:
    """Represents a learning strategy genome"""
    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.9, 
                 exploration_rate: float = 0.1, batch_size: int = 32):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.exploration_rate = exploration_rate
        self.batch_size = batch_size
        self.fitness = 0.0
        self.age = 0
        self.mutations = []
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate genome parameters"""
        mutations_applied = []
        
        if random.random() < mutation_rate:
            old_lr = self.learning_rate
            self.learning_rate *= random.uniform(0.8, 1.2)
            self.learning_rate = max(0.0001, min(0.01, self.learning_rate))
            mutations_applied.append(f"learning_rate: {old_lr:.5f} -> {self.learning_rate:.5f}")
        
        if random.random() < mutation_rate:
            old_momentum = self.momentum
            self.momentum *= random.uniform(0.9, 1.1)
            self.momentum = max(0.1, min(0.99, self.momentum))
            mutations_applied.append(f"momentum: {old_momentum:.3f} -> {self.momentum:.3f}")
        
        if random.random() < mutation_rate:
            old_explore = self.exploration_rate
            self.exploration_rate *= random.uniform(0.8, 1.2)
            self.exploration_rate = max(0.01, min(0.5, self.exploration_rate))
            mutations_applied.append(f"exploration: {old_explore:.3f} -> {self.exploration_rate:.3f}")
        
        self.mutations.extend(mutations_applied)
        return mutations_applied
    
    def crossover(self, other: 'LearningGenome') -> 'LearningGenome':
        """Create offspring genome through crossover"""
        offspring = LearningGenome()
        offspring.learning_rate = random.choice([self.learning_rate, other.learning_rate])
        offspring.momentum = random.choice([self.momentum, other.momentum])
        offspring.exploration_rate = random.choice([self.exploration_rate, other.exploration_rate])
        offspring.batch_size = random.choice([self.batch_size, other.batch_size])
        return offspring
    
    def copy(self) -> 'LearningGenome':
        """Create a copy of this genome"""
        genome = LearningGenome(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            exploration_rate=self.exploration_rate,
            batch_size=self.batch_size
        )
        genome.fitness = self.fitness
        genome.age = self.age
        return genome

class AdaptivePattern:
    """Represents an adaptive learning pattern discovered through evolution"""
    def __init__(self, pattern_type: str, description: str, effectiveness: float):
        self.pattern_type = pattern_type
        self.description = description
        self.effectiveness = effectiveness
        self.usage_count = 0
        self.discovered_at = datetime.now().isoformat()
    
    def apply(self, network, data):
        """Apply this pattern to the network"""
        self.usage_count += 1
        # Pattern application logic would go here
        pass

class GeneticLearning:
    """
    Genetic algorithm-based learning system
    Evolves learning strategies to mimic human-like adaptive patterns
    """
    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.population: List[LearningGenome] = []
        self.generation = 0
        self.best_genome = None
        self.fitness_history = deque(maxlen=100)
        self.discovered_patterns = []
        
        self.initialize_population()
    
    def initialize_population(self):
        """Create initial population with diverse genomes"""
        for i in range(self.population_size):
            genome = LearningGenome(
                learning_rate=random.uniform(0.0001, 0.01),
                momentum=random.uniform(0.5, 0.99),
                exploration_rate=random.uniform(0.05, 0.3),
                batch_size=random.choice([16, 32, 64, 128])
            )
            self.population.append(genome)
        
        print(f"[GENETIC] Initialized population of {self.population_size} genomes")
    
    def evaluate_fitness(self, genome: LearningGenome, network, X_train, y_train) -> float:
        """Evaluate genome fitness based on learning performance"""
        # Use genome parameters to train for a few iterations
        original_lr = network.weights[0].copy() if network.weights else None
        
        fitness = 0.0
        try:
            # Train with this genome's parameters
            batch_size = min(genome.batch_size, len(X_train))
            indices = np.random.choice(len(X_train), batch_size, replace=False)
            X_batch = X_train[indices]
            y_batch = y_train[indices]
            
            # Forward pass
            network.forward(X_batch, training=True)
            
            # Calculate accuracy
            predictions = network.predict(X_train)
            accuracy = float(np.mean(predictions == y_train))
            
            # Get confidence
            confidences = network.get_confidence(X_train)
            correct = predictions == y_train
            confidence = float(np.mean(confidences[correct]) if np.any(correct) else 0)
            
            # Fitness = accuracy weighted by confidence
            fitness = accuracy * 0.7 + confidence * 0.3
            
            # Bonus for exploration (diverse learning)
            fitness += genome.exploration_rate * 0.1
            
        except Exception as e:
            print(f"Fitness evaluation error: {e}")
            fitness = 0.0
        
        genome.fitness = fitness
        genome.age += 1
        return fitness
    
    def select_parents(self, tournament_size: int = 3) -> Tuple[LearningGenome, LearningGenome]:
        """Select two parents using tournament selection"""
        def tournament():
            candidates = random.sample(self.population, min(tournament_size, len(self.population)))
            return max(candidates, key=lambda g: g.fitness)
        
        parent1 = tournament()
        parent2 = tournament()
        return parent1, parent2
    
    def evolve_generation(self, network, X_train, y_train) -> Dict[str, Any]:
        """Evolve one generation of genomes"""
        self.generation += 1
        
        # Evaluate fitness for all genomes
        for genome in self.population:
            self.evaluate_fitness(genome, network, X_train, y_train)
        
        # Sort by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Track best genome
        current_best = self.population[0]
        if self.best_genome is None or current_best.fitness > self.best_genome.fitness:
            self.best_genome = current_best.copy()
            print(f"[GENETIC] New best genome! Fitness: {current_best.fitness:.4f}")
        
        self.fitness_history.append(current_best.fitness)
        
        # Create new population
        new_population = []
        
        # Elitism: keep top 20%
        elite_count = max(1, int(self.population_size * 0.2))
        new_population.extend([g.copy() for g in self.population[:elite_count]])
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            
            # Crossover
            if random.random() < 0.7:  # 70% crossover rate
                offspring = parent1.crossover(parent2)
            else:
                offspring = parent1.copy()
            
            # Mutation
            mutations = offspring.mutate(mutation_rate=0.15)
            if mutations:
                print(f"[GENETIC] Mutation: {', '.join(mutations)}")
            
            new_population.append(offspring)
        
        self.population = new_population
        
        # Detect adaptive patterns
        self._detect_adaptive_patterns()
        
        return {
            'generation': self.generation,
            'best_fitness': current_best.fitness,
            'avg_fitness': np.mean([g.fitness for g in self.population]),
            'population_diversity': self._calculate_diversity(),
            'patterns_discovered': len(self.discovered_patterns)
        }
    
    def _detect_adaptive_patterns(self):
        """Detect and record adaptive learning patterns"""
        # Pattern: Rapid learning (high learning rate + high fitness)
        rapid_learners = [g for g in self.population if g.learning_rate > 0.005 and g.fitness > 0.7]
        if rapid_learners and len(rapid_learners) > 2:
            pattern = AdaptivePattern(
                pattern_type='rapid_learning',
                description=f'Rapid learning pattern: {len(rapid_learners)} genomes with high LR and fitness',
                effectiveness=np.mean([g.fitness for g in rapid_learners])
            )
            self.discovered_patterns.append(pattern)
        
        # Pattern: Stable learning (low exploration + high fitness)
        stable_learners = [g for g in self.population if g.exploration_rate < 0.1 and g.fitness > 0.6]
        if stable_learners and len(stable_learners) > 2:
            pattern = AdaptivePattern(
                pattern_type='stable_learning',
                description=f'Stable learning pattern: {len(stable_learners)} genomes with low exploration and high fitness',
                effectiveness=np.mean([g.fitness for g in stable_learners])
            )
            self.discovered_patterns.append(pattern)
        
        # Pattern: Exploratory learning (high exploration + improving fitness)
        if len(self.fitness_history) >= 5:
            recent_improvement = self.fitness_history[-1] - self.fitness_history[-5]
            exploratory = [g for g in self.population if g.exploration_rate > 0.2]
            if exploratory and recent_improvement > 0.1:
                pattern = AdaptivePattern(
                    pattern_type='exploratory_learning',
                    description=f'Exploratory learning leads to improvement: {recent_improvement:.3f} gain',
                    effectiveness=recent_improvement
                )
                self.discovered_patterns.append(pattern)
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if len(self.population) < 2:
            return 0.0
        
        # Calculate variance in key parameters
        learning_rates = [g.learning_rate for g in self.population]
        momentums = [g.momentum for g in self.population]
        explorations = [g.exploration_rate for g in self.population]
        
        diversity = (
            np.var(learning_rates) +
            np.var(momentums) +
            np.var(explorations)
        ) / 3.0
        
        return float(diversity)
    
    def get_best_strategy(self) -> Dict[str, Any]:
        """Get the best learning strategy discovered"""
        if not self.best_genome:
            return {}
        
        return {
            'learning_rate': self.best_genome.learning_rate,
            'momentum': self.best_genome.momentum,
            'exploration_rate': self.best_genome.exploration_rate,
            'batch_size': self.best_genome.batch_size,
            'fitness': self.best_genome.fitness,
            'age': self.best_genome.age,
            'mutations': self.best_genome.mutations[-10:]
        }
    
    def apply_best_strategy_to_network(self, network) -> Dict[str, Any]:
        """Apply the best evolved strategy to the network"""
        if not self.best_genome:
            return {}
        
        strategy = self.get_best_strategy()
        
        # This would be used to guide the network's learning
        # For now, we return it for the trainer to use
        print(f"[GENETIC] Applying best strategy: LR={strategy['learning_rate']:.5f}, "
              f"Momentum={strategy['momentum']:.3f}, Exploration={strategy['exploration_rate']:.3f}")
        
        return strategy
    
    def get_adaptive_patterns_summary(self) -> Dict[str, Any]:
        """Get summary of discovered adaptive patterns"""
        pattern_counts = {}
        for pattern in self.discovered_patterns:
            pattern_counts[pattern.pattern_type] = pattern_counts.get(pattern.pattern_type, 0) + 1
        
        return {
            'total_patterns': len(self.discovered_patterns),
            'pattern_types': pattern_counts,
            'recent_patterns': [
                {
                    'type': p.pattern_type,
                    'description': p.description,
                    'effectiveness': p.effectiveness,
                    'discovered_at': p.discovered_at
                }
                for p in self.discovered_patterns[-10:]
            ],
            'most_effective': max(
                self.discovered_patterns,
                key=lambda p: p.effectiveness
            ).pattern_type if self.discovered_patterns else None
        }
