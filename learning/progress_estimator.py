
import time
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque

class AdvancedProgressEstimator:
    """GPS-style ETA estimator using Kalman filtering and statistical analysis"""
    
    def __init__(self, total_stages: int = 7, understanding_threshold: float = 0.999):
        self.total_stages = total_stages
        self.understanding_threshold = understanding_threshold
        
        # Persistence files
        self.state_file = 'estimator_state.json'
        self.history_file = 'training_history.json'
        self.last_visit_file = 'last_visit.json'
        
        # Historical data
        self.stage_history = []
        self.current_stage_data = {
            'start_time': None,
            'iterations': deque(maxlen=100),
            'understanding_scores': deque(maxlen=100),
            'timestamps': deque(maxlen=100),
            'stage_index': 0,
            'stage_name': ''
        }
        
        # Kalman filter state for velocity estimation
        self.kalman_state = {
            'velocity': 0.0,  # understanding units per second
            'velocity_variance': 1.0,
            'process_noise': 0.01,
            'measurement_noise': 0.05
        }
        
        # Exponential smoothing parameters
        self.alpha = 0.3  # smoothing factor for recent data
        self.beta = 0.1   # trend smoothing factor
        self.smoothed_velocity = 0.0
        self.velocity_trend = 0.0
        
        # Running statistics
        self.avg_iterations_per_stage = None
        self.avg_time_per_stage = None
        self.stage_completion_rates = []
        
        # Load previous state
        self.load_state()
    
    def start_stage(self, stage_index: int, stage_name: str):
        """Mark the start of a new stage"""
        self.current_stage_data = {
            'start_time': time.time(),
            'iterations': deque(maxlen=100),
            'understanding_scores': deque(maxlen=100),
            'timestamps': deque(maxlen=100),
            'stage_index': stage_index,
            'stage_name': stage_name,
            'last_update': time.time()
        }
        # Reset Kalman filter for new stage
        self.kalman_state['velocity'] = 0.0
        self.kalman_state['velocity_variance'] = 1.0
        self.smoothed_velocity = 0.0
        self.velocity_trend = 0.0
    
    def update_progress(self, iteration: int, understanding: float):
        """Update progress with Kalman filtering"""
        if self.current_stage_data['start_time'] is None:
            return
        
        current_time = time.time()
        self.current_stage_data['iterations'].append(iteration)
        self.current_stage_data['understanding_scores'].append(understanding)
        self.current_stage_data['timestamps'].append(current_time)
        self.current_stage_data['last_update'] = current_time
        
        # Update Kalman filter
        if len(self.current_stage_data['understanding_scores']) >= 2:
            self._update_kalman_filter()
        
        # Auto-save every 50 iterations
        if iteration % 50 == 0:
            self.save_state()
    
    def _update_kalman_filter(self):
        """Update Kalman filter for velocity estimation"""
        scores = list(self.current_stage_data['understanding_scores'])
        times = list(self.current_stage_data['timestamps'])
        
        if len(scores) < 2:
            return
        
        # Measured velocity (recent change)
        dt = times[-1] - times[-2]
        if dt > 0:
            measured_velocity = (scores[-1] - scores[-2]) / dt
            
            # Kalman filter prediction
            predicted_velocity = self.kalman_state['velocity']
            predicted_variance = self.kalman_state['velocity_variance'] + self.kalman_state['process_noise']
            
            # Kalman gain
            kalman_gain = predicted_variance / (predicted_variance + self.kalman_state['measurement_noise'])
            
            # Update
            self.kalman_state['velocity'] = predicted_velocity + kalman_gain * (measured_velocity - predicted_velocity)
            self.kalman_state['velocity_variance'] = (1 - kalman_gain) * predicted_variance
            
            # Exponential smoothing for additional stability
            old_smoothed = self.smoothed_velocity
            self.smoothed_velocity = self.alpha * self.kalman_state['velocity'] + (1 - self.alpha) * self.smoothed_velocity
            self.velocity_trend = self.beta * (self.smoothed_velocity - old_smoothed) + (1 - self.beta) * self.velocity_trend
    
    def complete_stage(self, final_iteration: int, final_understanding: float):
        """Mark stage completion and update statistics"""
        if self.current_stage_data['start_time'] is None:
            return
        
        elapsed_time = time.time() - self.current_stage_data['start_time']
        
        stage_record = {
            'stage_index': self.current_stage_data['stage_index'],
            'stage_name': self.current_stage_data['stage_name'],
            'iterations': final_iteration,
            'duration_seconds': elapsed_time,
            'final_understanding': final_understanding,
            'avg_velocity': self.smoothed_velocity,
            'timestamp': datetime.now().isoformat()
        }
        
        self.stage_history.append(stage_record)
        self.stage_completion_rates.append(final_understanding / elapsed_time if elapsed_time > 0 else 0)
        
        # Update running averages with exponential weighting (recent stages matter more)
        if self.stage_history:
            weights = np.exp(np.linspace(-1, 0, len(self.stage_history)))
            weights /= weights.sum()
            
            self.avg_iterations_per_stage = np.average(
                [s['iterations'] for s in self.stage_history],
                weights=weights
            )
            self.avg_time_per_stage = np.average(
                [s['duration_seconds'] for s in self.stage_history],
                weights=weights
            )
    
    def estimate_current_stage_completion(self) -> Optional[Dict]:
        """GPS-style ETA using multiple statistical models"""
        scores = list(self.current_stage_data['understanding_scores'])
        times = list(self.current_stage_data['timestamps'])
        
        if len(scores) < 5:
            return None
        
        current_understanding = scores[-1]
        current_time = times[-1]
        
        # Already complete
        if current_understanding >= self.understanding_threshold:
            return {
                'eta_seconds': 0,
                'eta_formatted': 'Complete!',
                'confidence': 1.0,
                'method': 'threshold_met'
            }
        
        remaining_distance = self.understanding_threshold - current_understanding
        
        # Model 1: Kalman-filtered velocity projection
        kalman_eta = self._estimate_kalman_eta(remaining_distance)
        
        # Model 2: Polynomial regression (accounts for acceleration/deceleration)
        poly_eta = self._estimate_polynomial_eta(scores, times, remaining_distance)
        
        # Model 3: Moving average with trend
        ma_eta = self._estimate_moving_average_eta(scores, times, remaining_distance)
        
        # Model 4: Historical pattern matching
        hist_eta = self._estimate_historical_eta(current_understanding, current_time)
        
        # Ensemble: Weighted combination of models
        models = []
        if kalman_eta is not None:
            models.append(('kalman', kalman_eta, 0.35))
        if poly_eta is not None:
            models.append(('polynomial', poly_eta, 0.30))
        if ma_eta is not None:
            models.append(('moving_avg', ma_eta, 0.25))
        if hist_eta is not None:
            models.append(('historical', hist_eta, 0.10))
        
        if not models:
            return None
        
        # Weighted ensemble
        total_weight = sum(w for _, _, w in models)
        ensemble_eta = sum(eta * w for _, eta, w in models) / total_weight
        
        # Calculate confidence based on model agreement
        model_etas = [eta for _, eta, _ in models]
        std_dev = np.std(model_etas)
        mean_eta = np.mean(model_etas)
        coefficient_of_variation = std_dev / (mean_eta + 1e-6)
        confidence = max(0.3, min(0.95, 1.0 - coefficient_of_variation))
        
        return {
            'eta_seconds': max(0, ensemble_eta),
            'eta_formatted': self._format_time(ensemble_eta),
            'confidence': confidence,
            'method': 'ensemble',
            'model_details': {m[0]: m[1] for m in models}
        }
    
    def _estimate_kalman_eta(self, remaining_distance: float) -> Optional[float]:
        """ETA using Kalman-filtered velocity with trend"""
        if self.smoothed_velocity <= 0:
            return None
        
        # Project velocity with trend
        projected_velocity = self.smoothed_velocity + self.velocity_trend
        
        if projected_velocity <= 0:
            projected_velocity = self.smoothed_velocity
        
        if projected_velocity > 0:
            return remaining_distance / projected_velocity
        return None
    
    def _estimate_polynomial_eta(self, scores: List[float], times: List[float], remaining: float) -> Optional[float]:
        """ETA using polynomial regression"""
        if len(scores) < 10:
            return None
        
        # Fit 2nd degree polynomial to recent data
        recent_window = min(30, len(scores))
        x = np.array(times[-recent_window:]) - times[-recent_window]
        y = np.array(scores[-recent_window:])
        
        try:
            coeffs = np.polyfit(x, y, 2)
            poly = np.poly1d(coeffs)
            
            # Find when polynomial reaches threshold
            current_time = times[-1]
            for future_offset in np.linspace(0, 3600, 1000):  # Search up to 1 hour
                predicted = poly(x[-1] + future_offset)
                if predicted >= self.understanding_threshold:
                    return future_offset
            
            # Fallback: linear extrapolation from derivative
            derivative = np.polyder(poly)
            current_velocity = derivative(x[-1])
            if current_velocity > 0:
                return remaining / current_velocity
        except:
            pass
        
        return None
    
    def _estimate_moving_average_eta(self, scores: List[float], times: List[float], remaining: float) -> Optional[float]:
        """ETA using exponentially weighted moving average"""
        if len(scores) < 5:
            return None
        
        # Calculate weighted velocities
        velocities = []
        for i in range(1, len(scores)):
            dt = times[i] - times[i-1]
            if dt > 0:
                velocities.append((scores[i] - scores[i-1]) / dt)
        
        if not velocities:
            return None
        
        # Exponential weights (recent data more important)
        weights = np.exp(np.linspace(-1, 0, len(velocities)))
        weights /= weights.sum()
        
        weighted_velocity = np.average(velocities, weights=weights)
        
        if weighted_velocity > 0:
            return remaining / weighted_velocity
        return None
    
    def _estimate_historical_eta(self, current_understanding: float, current_time: float) -> Optional[float]:
        """ETA based on historical completion patterns"""
        if not self.stage_history or self.current_stage_data['start_time'] is None:
            return None
        
        elapsed = current_time - self.current_stage_data['start_time']
        progress_ratio = current_understanding / self.understanding_threshold
        
        if progress_ratio <= 0:
            return None
        
        # Estimate total time based on progress ratio
        estimated_total_time = elapsed / progress_ratio
        remaining_time = estimated_total_time - elapsed
        
        # Adjust based on historical average if available
        if self.avg_time_per_stage is not None:
            # Blend with historical data
            blended_total = 0.6 * estimated_total_time + 0.4 * self.avg_time_per_stage
            remaining_time = blended_total - elapsed
        
        return max(0, remaining_time)
    
    def estimate_total_completion(self) -> Optional[Dict]:
        """Estimate total time to complete all stages"""
        current_stage_eta = self.estimate_current_stage_completion()
        
        if current_stage_eta is None:
            return None
        
        current_stage_time = current_stage_eta['eta_seconds']
        stages_remaining = self.total_stages - self.current_stage_data['stage_index'] - 1
        
        if stages_remaining <= 0:
            return {
                'eta_seconds': current_stage_time,
                'eta_formatted': self._format_time(current_stage_time),
                'confidence': current_stage_eta['confidence'],
                'stages_remaining': 0
            }
        
        # Use historical data with stage difficulty scaling
        if self.avg_time_per_stage and len(self.stage_history) >= 2:
            # Calculate difficulty trend (are stages getting harder?)
            recent_times = [s['duration_seconds'] for s in self.stage_history[-3:]]
            if len(recent_times) >= 2:
                difficulty_trend = np.polyfit(range(len(recent_times)), recent_times, 1)[0]
            else:
                difficulty_trend = 0
            
            # Project future stage times with trend
            future_stage_times = []
            for i in range(stages_remaining):
                projected_time = self.avg_time_per_stage + difficulty_trend * (i + 1)
                future_stage_times.append(max(self.avg_time_per_stage * 0.5, projected_time))
            
            remaining_stages_time = sum(future_stage_times)
            confidence = min(0.85, 0.5 + len(self.stage_history) * 0.05)
        else:
            # Fallback: assume stages get progressively harder
            current_elapsed = time.time() - self.current_stage_data['start_time']
            current_understanding = list(self.current_stage_data['understanding_scores'])[-1] if self.current_stage_data['understanding_scores'] else 0.1
            
            estimated_current_total = current_elapsed / max(0.1, current_understanding)
            remaining_stages_time = estimated_current_total * stages_remaining * 1.3  # 30% harder each stage
            confidence = 0.4
        
        total_time = current_stage_time + remaining_stages_time
        
        return {
            'eta_seconds': total_time,
            'eta_formatted': self._format_time(total_time),
            'confidence': confidence,
            'stages_remaining': stages_remaining,
            'current_stage_eta_seconds': current_stage_time
        }
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time"""
        if seconds <= 0:
            return '0s'
        
        seconds = int(seconds)
        parts = []
        
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if seconds or not parts:
            parts.append(f"{seconds}s")
        
        return ' '.join(parts)
    
    def get_stats(self) -> Dict:
        """Get estimator statistics"""
        return {
            'completed_stages': len(self.stage_history),
            'current_stage': self.current_stage_data['stage_name'],
            'avg_iterations_per_stage': self.avg_iterations_per_stage,
            'avg_time_per_stage': self.avg_time_per_stage,
            'current_velocity': self.smoothed_velocity,
            'velocity_trend': self.velocity_trend,
            'total_elapsed_time': sum(s['duration_seconds'] for s in self.stage_history) + 
                                 (time.time() - self.current_stage_data['start_time'] if self.current_stage_data['start_time'] else 0),
            'history': self.stage_history
        }
    
    def save_state(self):
        """Save estimator state to files"""
        state = {
            'stage_history': self.stage_history,
            'current_stage': {
                'start_time': self.current_stage_data['start_time'],
                'iterations': list(self.current_stage_data['iterations']),
                'understanding_scores': list(self.current_stage_data['understanding_scores']),
                'timestamps': list(self.current_stage_data['timestamps']),
                'stage_index': self.current_stage_data['stage_index'],
                'stage_name': self.current_stage_data['stage_name'],
                'last_update': self.current_stage_data['last_update']
            },
            'avg_iterations_per_stage': self.avg_iterations_per_stage,
            'avg_time_per_stage': self.avg_time_per_stage,
            'kalman_state': self.kalman_state,
            'smoothed_velocity': self.smoothed_velocity,
            'velocity_trend': self.velocity_trend,
            'understanding_threshold': self.understanding_threshold,
            'last_save': datetime.now().isoformat()
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save last visit marker
        if self.current_stage_data['understanding_scores']:
            last_visit = {
                'iteration': int(list(self.current_stage_data['iterations'])[-1]) if self.current_stage_data['iterations'] else 0,
                'understanding': float(list(self.current_stage_data['understanding_scores'])[-1]),
                'stage': self.current_stage_data['stage_name'],
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.last_visit_file, 'w') as f:
                json.dump(last_visit, f, indent=2)
    
    def load_state(self):
        """Load estimator state from files"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.stage_history = state.get('stage_history', [])
                self.avg_iterations_per_stage = state.get('avg_iterations_per_stage')
                self.avg_time_per_stage = state.get('avg_time_per_stage')
                self.understanding_threshold = state.get('understanding_threshold', 0.999)
                
                if 'kalman_state' in state:
                    self.kalman_state = state['kalman_state']
                if 'smoothed_velocity' in state:
                    self.smoothed_velocity = state['smoothed_velocity']
                if 'velocity_trend' in state:
                    self.velocity_trend = state['velocity_trend']
                
                print(f"✓ Loaded estimator state: {len(self.stage_history)} completed stages")
            except Exception as e:
                print(f"Could not load state: {e}")
        
        # Load last visit data
        if os.path.exists(self.last_visit_file):
            try:
                with open(self.last_visit_file, 'r') as f:
                    last_visit_data = json.load(f)
                
                print(f"✓ Last visit: Iteration {last_visit_data.get('iteration', 0)}, "
                      f"Understanding {last_visit_data.get('understanding', 0):.2%}")
            except Exception as e:
                print(f"Could not load last visit: {e}")

# Backwards compatibility alias
ProgressEstimator = AdvancedProgressEstimator
