from typing import Optional, Dict, Any
import time
from datetime import datetime
import threading
from .learning_optimizer import LearningOptimizer
from .network_optimizer import NetworkOptimizer

class OptimizationIntegrator:
    def __init__(self):
        self.learning_optimizer = LearningOptimizer(base_learning_rate=0.001)
        self.network_optimizer = NetworkOptimizer(host='0.0.0.0', port=5000)
        self._start_time = time.time()

    def start(self):
        """Start optimization monitoring"""
        self.network_optimizer.start_monitoring()

    def stop(self):
        """Stop optimization monitoring"""
        self.network_optimizer.stop_monitoring()

    def optimize_training(self, current_accuracy: float, 
                         gradients: Optional[Any] = None):
        """Optimize training parameters"""
        # Get optimized learning rate
        learning_rate = self.learning_optimizer.optimize_learning_rate(
            current_accuracy
        )

        # Apply momentum if gradients provided
        if gradients is not None:
            gradients = self.learning_optimizer.apply_momentum(gradients)

        return {
            'learning_rate': learning_rate,
            'optimized_gradients': gradients,
            'stats': self.learning_optimizer.get_learning_stats()
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'uptime_seconds': time.time() - self._start_time,
            'learning_stats': self.learning_optimizer.get_learning_stats(),
            'network_stats': self.network_optimizer.get_network_stats()
        }