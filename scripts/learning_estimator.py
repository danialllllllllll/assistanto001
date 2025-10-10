import numpy as np
from typing import List, Tuple, Optional
import json
import time
from datetime import datetime

class EnhancedEstimator:
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.metrics_history = []
        self.stage_thresholds = {
            'Baby Steps': 0.7,
            'Toddler': 0.8,
            'Pre-K': 0.85,
            'Elementary': 0.9,
            'Teen': 0.93,
            'Scholar': 0.96,
            'Thinker': 0.999
        }

    def add_metrics(self, understanding: float, confidence: float, 
                   accuracy: float, timestamp: float):
        """Add new metrics to history"""
        self.metrics_history.append({
            'understanding': understanding,
            'confidence': confidence,
            'accuracy': accuracy,
            'timestamp': timestamp
        })

        if len(self.metrics_history) > self.history_size:
            self.metrics_history = self.metrics_history[-self.history_size:]

    def estimate_completion(self, current_stage: str) -> dict:
        """Estimate completion time using advanced statistical analysis"""
        if len(self.metrics_history) < 2:
            return {
                'eta_seconds': None,
                'confidence': 0,
                'trend': 'insufficient_data'
            }

        target = self.stage_thresholds.get(current_stage, 0.999)
        window_sizes = [5, 10, 20, 50]
        trends = []

        for window in window_sizes:
            if len(self.metrics_history) >= window:
                recent = self.metrics_history[-window:]
                understanding_values = [m['understanding'] for m in recent]
                trend = np.polyfit(range(len(recent)), 
                                 understanding_values, 1)[0]
                trends.append(trend)

        if not trends:
            return {
                'eta_seconds': None,
                'confidence': 0,
                'trend': 'insufficient_data'
            }

        weighted_trend = np.average(trends, 
                                  weights=range(1, len(trends) + 1))

        current = self.metrics_history[-1]['understanding']
        distance_to_target = target - current

        if weighted_trend <= 0:
            return {
                'eta_seconds': None,
                'confidence': 0,
                'trend': 'negative'
            }

        eta_seconds = distance_to_target / weighted_trend
        trend_consistency = np.std(trends) / np.mean(trends)
        confidence = 1.0 - min(1.0, trend_consistency)

        return {
            'eta_seconds': int(eta_seconds),
            'confidence': confidence,
            'trend': 'positive' if weighted_trend > 0 else 'neutral'
        }