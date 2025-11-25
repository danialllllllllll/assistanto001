from typing import Dict, Any, Optional, List, Union, TypedDict
import sys
from pathlib import Path
import importlib.util
from datetime import datetime
import numpy as np

class TrainingMetrics(TypedDict):
    accuracy: float
    understanding_score: float
    confidence: float
    iteration: int
    timestamp: str

class TrainingBridge:
    """
    Bridge class to make training system accessible and type-safe
    """
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self._setup_paths()
        self.metrics = UnderstandingMetrics()
        self.current_metrics = {
            'accuracy': 0.0,
            'understanding_score': 0.0,
            'confidence': 0.0,
            'iteration': 0
        }

    def get_current_metrics(self) -> TrainingMetrics:
        """Get current training metrics with proper typing"""
        return {
            'accuracy': float(self.current_metrics['accuracy']),
            'understanding_score': float(self.current_metrics['understanding_score']),
            'confidence': float(self.current_metrics['confidence']),
            'iteration': int(self.current_metrics['iteration']),
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        }

    def update_metrics(self, network, X, y) -> TrainingMetrics:
        """
        Update current metrics using the training data
        """
        predictions = network.predict(X)
        confidences = network.get_confidence(X)

        correct = predictions == y
        self.current_metrics.update({
            'accuracy': float(np.mean(correct)),
            'confidence': float(np.mean(confidences)),
            'iteration': self.current_metrics['iteration'] + 1
        })

        # Calculate understanding score
        understanding_metrics = self.metrics.calculate_understanding_score(
            network=network,
            X=X,
            y=y
        )

        self.current_metrics['understanding_score'] = float(
            understanding_metrics['understanding']
        )

        return self.get_current_metrics()
