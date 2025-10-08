import numpy as np
import json
from core.neural_network import ProgressiveNeuralNetwork
from core.metrics import UnderstandingMetrics

class UnderstandingTrainer:
    """Gradient descent-based trainer optimized for 99.9% understanding + confidence"""
    
    def __init__(self, network, stage_config):
        self.network = network
        self.stage_config = stage_config
        self.metrics_calculator = UnderstandingMetrics()
        self.training_history = []
    
    def train_stage(self, X, y, stage_info, verbose=True):
        """
        Train until 99.9% understanding threshold is met
        Uses gradient descent with adaptive learning rate
        """
        stage_name = stage_info['name']
        learning_rate = stage_info['learning_rate']
        max_iterations = stage_info['max_iterations']
        understanding_threshold = stage_info['understanding_threshold']
        
        self.network.set_stage_activation(stage_info['active_nodes_percent'])
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"STAGE: {stage_name}")
            print(f"{'='*60}")
            print(f"{stage_info['description']}")
            print(f"Active nodes: {stage_info['active_nodes_percent']*100:.0f}%")
            print(f"Understanding threshold: {understanding_threshold*100:.1f}%")
            print(f"{'='*60}\n")
        
        best_understanding = 0
        patience = 50
        patience_counter = 0
        adaptive_lr = learning_rate
        iteration = 0
        
        while True:
            batch_size = min(64, len(X))
            indices = np.random.choice(len(X), batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            
            self.network.forward(X_batch)
            self.network.backward(X_batch, y_batch, adaptive_lr)
            
            if iteration % 10 == 0:
                metrics = self.metrics_calculator.calculate_understanding_score(
                    self.network, X, y
                )
                
                if verbose and iteration % 50 == 0:
                    print(f"Iteration {iteration:4d}: "
                          f"Understanding={metrics['understanding']:.4f}, "
                          f"Accuracy={metrics['raw_accuracy']:.4f}, "
                          f"Confidence={metrics['confidence']:.4f}")
                
                if verbose and iteration % 1 == 0:
                    print(f"Iteration {iteration} reached. Continuing until 99.9% understanding...")
                
                if metrics['understanding'] > best_understanding:
                    best_understanding = metrics['understanding']
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if self.metrics_calculator.check_threshold(
                    metrics, understanding_threshold, 0.999
                ):
                    if verbose:
                        print(f"\n✓ Understanding threshold achieved at iteration {iteration}!")
                        print(f"  Understanding: {metrics['understanding']:.4f}")
                        print(f"  Confidence: {metrics['confidence']:.4f}")
                    break
                
                if patience_counter >= patience:
                    adaptive_lr *= 0.5
                    patience_counter = 0
                    if verbose and adaptive_lr >= learning_rate * 0.01:
                        print(f"  Reducing learning rate to {adaptive_lr:.6f}")
            
            iteration += 1
        
        final_metrics = self.metrics_calculator.calculate_understanding_score(
            self.network, X, y
        )
        
        stage_report = self.metrics_calculator.get_stage_report(stage_name, final_metrics)
        self.training_history.append(stage_report)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"STAGE COMPLETE: {stage_name}")
            print(f"Understanding Score: {final_metrics['understanding']:.4f} (threshold: {understanding_threshold:.3f})")
            print(f"Accuracy: {final_metrics['raw_accuracy']:.4f}")
            print(f"Confidence: {final_metrics['confidence']:.4f}")
            print(f"Status: {'PASSED ✓' if stage_report['passed'] else 'NEEDS MORE WORK ✗'}")
            print(f"{'='*60}\n")
            
            if not stage_report['passed']:
                print(f"⚠ Stage {stage_name} did not reach 99.9% threshold.")
                print(f"  Understanding achieved: {final_metrics['understanding']*100:.2f}%")
                print(f"  Required: 99.90%")
                print()
        
        return stage_report
    
    def train_all_stages(self, X, y, stages, enforce_threshold=True):
        """Train through all developmental stages"""
        print("\n" + "="*60)
        print("UNDERSTANDING-FOCUSED AI TRAINING")
        print("Quality Over Quantity - 99.9% Understanding Required")
        print("="*60)
        
        for stage in stages:
            stage_report = self.train_stage(X, y, stage)
            
            if enforce_threshold and not stage_report['passed']:
                print(f"\n{'!'*60}")
                print(f"CRITICAL: Stage {stage['name']} did not meet 99.9% threshold!")
                print(f"Cannot proceed to next stage with strict enforcement enabled.")
                print(f"{'!'*60}\n")
                break
        
        return self.training_history
    
    def get_training_summary(self):
        """Get summary of training progress"""
        return {
            'stages_completed': len(self.training_history),
            'history': self.training_history,
            'all_passed': all(stage['passed'] for stage in self.training_history)
        }
