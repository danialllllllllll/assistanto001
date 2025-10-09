import numpy as np
from typing import Dict, List, Tuple, Any
import threading
import time

class LearningOptimizer:
    def __init__(self, base_learning_rate: float = 0.001):
        self.base_learning_rate = base_learning_rate
        self.learning_stats = {
            'iterations': 0,
            'improvements': 0,
            'last_accuracy': 0.0,
            'best_accuracy': 0.0,
            'stagnant_iterations': 0
        }
        self.adaptive_rate = base_learning_rate
        self.momentum = 0.9
        self.previous_gradients = None
        self.lock = threading.Lock()

    def optimize_learning_rate(self, current_accuracy: float) -> float:
        """Dynamically adjust learning rate based on training progress"""
        with self.lock:
            self.learning_stats['iterations'] += 1

            # Check for improvement
            if current_accuracy > self.learning_stats['last_accuracy']:
                self.learning_stats['improvements'] += 1
                self.learning_stats['stagnant_iterations'] = 0

                # If consistently improving, gradually increase learning rate
                if self.learning_stats['improvements'] > 5:
                    self.adaptive_rate *= 1.1
            else:
                self.learning_stats['stagnant_iterations'] += 1

                # If stagnating, reduce learning rate
                if self.learning_stats['stagnant_iterations'] > 3:
                    self.adaptive_rate *= 0.5
                    self.learning_stats['stagnant_iterations'] = 0

            # Keep learning rate within reasonable bounds
            self.adaptive_rate = max(min(self.adaptive_rate, 0.1), 0.0001)

            # Update tracking metrics
            self.learning_stats['last_accuracy'] = current_accuracy
            self.learning_stats['best_accuracy'] = max(
                self.learning_stats['best_accuracy'], 
                current_accuracy
            )

            return self.adaptive_rate

    def apply_momentum(self, gradients: np.ndarray) -> np.ndarray:
        """Apply momentum to gradients for more stable learning"""
        if self.previous_gradients is None:
            self.previous_gradients = gradients
        else:
            gradients = (self.momentum * self.previous_gradients + 
                        (1 - self.momentum) * gradients)
            self.previous_gradients = gradients
        return gradients

    def get_learning_stats(self) -> Dict[str, Any]:
        """Return current learning statistics"""
        with self.lock:
            return {
                'current_learning_rate': self.adaptive_rate,
                'base_learning_rate': self.base_learning_rate,
                'iterations': self.learning_stats['iterations'],
                'improvements': self.learning_stats['improvements'],
                'best_accuracy': self.learning_stats['best_accuracy'],
                'stagnant_iterations': self.learning_stats['stagnant_iterations']
            }