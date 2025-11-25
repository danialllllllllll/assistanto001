"""
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
