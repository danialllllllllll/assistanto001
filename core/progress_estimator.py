import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class ProgressEstimator:
    """Functional estimator for learning progress based on actual iteration data"""
    
    def __init__(self, total_stages: int = 7, understanding_threshold: float = 0.999):
        self.total_stages = total_stages
        self.understanding_threshold = understanding_threshold
        
        # Historical data
        self.stage_history = []
        self.current_stage_data = {
            'start_time': None,
            'iterations': [],
            'understanding_scores': [],
            'stage_index': 0,
            'stage_name': ''
        }
        
        # Running averages
        self.avg_iterations_per_stage = None
        self.avg_time_per_stage = None
    
    def start_stage(self, stage_index: int, stage_name: str):
        """Mark the start of a new stage"""
        self.current_stage_data = {
            'start_time': time.time(),
            'iterations': [],
            'understanding_scores': [],
            'stage_index': stage_index,
            'stage_name': stage_name,
            'last_update': time.time()
        }
    
    def update_progress(self, iteration: int, understanding: float):
        """Update progress for current stage"""
        if self.current_stage_data['start_time'] is None:
            return
        
        self.current_stage_data['iterations'].append(iteration)
        self.current_stage_data['understanding_scores'].append(understanding)
        self.current_stage_data['last_update'] = time.time()
    
    def complete_stage(self, final_iteration: int, final_understanding: float):
        """Mark stage completion and update historical averages"""
        if self.current_stage_data['start_time'] is None:
            return
        
        elapsed_time = time.time() - self.current_stage_data['start_time']
        
        stage_record = {
            'stage_index': self.current_stage_data['stage_index'],
            'stage_name': self.current_stage_data['stage_name'],
            'iterations': final_iteration,
            'duration_seconds': elapsed_time,
            'final_understanding': final_understanding,
            'timestamp': datetime.now().isoformat()
        }
        
        self.stage_history.append(stage_record)
        
        # Update running averages
        if self.stage_history:
            self.avg_iterations_per_stage = sum(s['iterations'] for s in self.stage_history) / len(self.stage_history)
            self.avg_time_per_stage = sum(s['duration_seconds'] for s in self.stage_history) / len(self.stage_history)
    
    def estimate_current_stage_completion(self) -> Optional[Dict]:
        """Estimate time to complete current stage using exponential smoothing"""
        if not self.current_stage_data['understanding_scores'] or len(self.current_stage_data['understanding_scores']) < 5:
            return None
        
        current_understanding = self.current_stage_data['understanding_scores'][-1]
        
        # If already at threshold, return 0
        if current_understanding >= self.understanding_threshold:
            return {
                'eta_seconds': 0,
                'eta_formatted': 'Complete!',
                'confidence': 1.0,
                'method': 'threshold_met'
            }
        
        # Calculate improvement rate using recent data points
        recent_window = min(20, len(self.current_stage_data['understanding_scores']))
        recent_scores = self.current_stage_data['understanding_scores'][-recent_window:]
        recent_iterations = self.current_stage_data['iterations'][-recent_window:]
        
        if len(recent_scores) < 2:
            return None
        
        # Calculate average improvement per iteration
        score_improvements = [recent_scores[i] - recent_scores[i-1] 
                            for i in range(1, len(recent_scores))]
        avg_improvement_per_iter = sum(score_improvements) / len(score_improvements) if score_improvements else 0
        
        # Calculate time per iteration
        elapsed_time = time.time() - self.current_stage_data['start_time']
        current_iteration = self.current_stage_data['iterations'][-1]
        time_per_iteration = elapsed_time / current_iteration if current_iteration > 0 else 1.0
        
        # If improving, calculate estimated iterations needed
        if avg_improvement_per_iter > 0.0001:
            remaining_understanding = self.understanding_threshold - current_understanding
            estimated_iterations_needed = remaining_understanding / avg_improvement_per_iter
            estimated_seconds = estimated_iterations_needed * time_per_iteration
            
            # Apply confidence based on consistency of improvement
            score_variance = sum((x - avg_improvement_per_iter) ** 2 for x in score_improvements) / len(score_improvements)
            confidence = max(0.3, min(0.95, 1.0 / (1.0 + score_variance * 100)))
            
            return {
                'eta_seconds': max(0, estimated_seconds),
                'eta_formatted': self._format_time(estimated_seconds),
                'confidence': confidence,
                'method': 'improvement_rate',
                'iterations_remaining': int(estimated_iterations_needed)
            }
        else:
            # No improvement detected, use historical average if available
            if self.avg_iterations_per_stage and self.avg_time_per_stage:
                remaining_time = max(0, self.avg_time_per_stage - elapsed_time)
                return {
                    'eta_seconds': remaining_time,
                    'eta_formatted': self._format_time(remaining_time),
                    'confidence': 0.5,
                    'method': 'historical_average'
                }
            
            return None
    
    def estimate_total_completion(self) -> Optional[Dict]:
        """Estimate total time to complete all remaining stages"""
        current_stage_eta = self.estimate_current_stage_completion()
        
        if current_stage_eta is None:
            return None
        
        # Time for current stage
        current_stage_time = current_stage_eta['eta_seconds']
        
        # Estimate remaining stages
        stages_remaining = self.total_stages - self.current_stage_data['stage_index'] - 1
        
        if stages_remaining <= 0:
            return {
                'eta_seconds': current_stage_time,
                'eta_formatted': self._format_time(current_stage_time),
                'confidence': current_stage_eta['confidence'],
                'stages_remaining': 0
            }
        
        # Use historical average if available, otherwise use current stage as reference
        if self.avg_time_per_stage and len(self.stage_history) > 0:
            remaining_stages_time = self.avg_time_per_stage * stages_remaining
            confidence = min(0.9, 0.5 + (len(self.stage_history) * 0.1))
        else:
            # Estimate: each stage takes progressively longer (stages get harder)
            elapsed_current = time.time() - self.current_stage_data['start_time']
            estimated_current_total = elapsed_current / max(0.1, self.current_stage_data['understanding_scores'][-1]) if self.current_stage_data['understanding_scores'] else elapsed_current * 2
            
            # Assume later stages take 1.2x as long on average
            remaining_stages_time = estimated_current_total * stages_remaining * 1.2
            confidence = 0.3
        
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
            'total_elapsed_time': sum(s['duration_seconds'] for s in self.stage_history) + 
                                 (time.time() - self.current_stage_data['start_time'] if self.current_stage_data['start_time'] else 0),
            'history': self.stage_history
        }
