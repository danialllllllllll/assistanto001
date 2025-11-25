import numpy as np

class UnderstandingMetrics:
    """Calculate understanding score based on accuracy, confidence, and consistency"""
    
    def __init__(self, min_confidence=0.999):
        self.min_confidence = min_confidence
        self.history = []
    
    def calculate_calibrated_accuracy(self, predictions, targets, confidences):
        """Calculate accuracy weighted by confidence calibration"""
        correct = predictions == targets
        accuracy = np.mean(correct)
        
        correct_confidences = confidences[correct]
        incorrect_confidences = confidences[~correct]
        
        avg_correct_conf = np.mean(correct_confidences) if len(correct_confidences) > 0 else 0
        avg_incorrect_conf = np.mean(incorrect_confidences) if len(incorrect_confidences) > 0 else 0
        
        calibration_score = avg_correct_conf - avg_incorrect_conf
        calibrated_accuracy = accuracy * (0.7 + 0.3 * max(0, calibration_score))
        
        return calibrated_accuracy, accuracy
    
    def calculate_confidence_score(self, confidences, predictions, targets):
        """Calculate confidence score with penalty for overconfidence on wrong answers"""
        correct = predictions == targets
        
        correct_conf = np.mean(confidences[correct]) if np.any(correct) else 0
        incorrect_conf = np.mean(confidences[~correct]) if np.any(~correct) else 0
        
        overconfidence_penalty = incorrect_conf * 0.5
        
        confidence_score = correct_conf - overconfidence_penalty
        
        return max(0, confidence_score)
    
    def calculate_consistency(self, predictions, confidences):
        """Check consistency in predictions and confidence levels"""
        if len(self.history) < 2:
            return 1.0
        
        prev_predictions = self.history[-1]['predictions']
        
        if len(prev_predictions) != len(predictions):
            return 0.5
        
        agreement = np.mean(predictions == prev_predictions)
        
        conf_stability = 1.0 - np.std(confidences)
        
        consistency = 0.7 * agreement + 0.3 * conf_stability
        
        return consistency
    
    def calculate_understanding_score(self, network, X, y):
        """
        Calculate comprehensive understanding score
        Understanding = calibrated_accuracy * 0.5 + confidence * 0.3 + consistency * 0.2
        """
        predictions = network.predict(X)
        confidences = network.get_confidence(X)
        
        calibrated_acc, raw_acc = self.calculate_calibrated_accuracy(predictions, y, confidences)
        confidence_score = self.calculate_confidence_score(confidences, predictions, y)
        consistency_score = self.calculate_consistency(predictions, confidences)
        
        understanding = (
            calibrated_acc * 0.5 +
            confidence_score * 0.3 +
            consistency_score * 0.2
        )
        
        avg_confidence = np.mean(confidences[predictions == y]) if np.any(predictions == y) else 0
        
        metrics = {
            'understanding': understanding,
            'calibrated_accuracy': calibrated_acc,
            'raw_accuracy': raw_acc,
            'confidence': avg_confidence,
            'confidence_score': confidence_score,
            'consistency': consistency_score,
            'predictions': predictions,
            'confidences': confidences
        }
        
        self.history.append(metrics)
        
        return metrics
    
    def check_threshold(self, metrics, min_understanding=0.999, min_confidence=0.999):
        """Check if understanding and confidence meet thresholds"""
        understanding_met = metrics['understanding'] >= min_understanding
        confidence_met = metrics['confidence'] >= min_confidence
        
        return understanding_met and confidence_met
    
    def get_stage_report(self, stage_name, metrics):
        """Generate detailed report for a stage"""
        return {
            'stage': stage_name,
            'understanding_score': float(metrics['understanding']),
            'accuracy': float(metrics['raw_accuracy']),
            'calibrated_accuracy': float(metrics['calibrated_accuracy']),
            'confidence': float(metrics['confidence']),
            'consistency': float(metrics['consistency']),
            'passed': metrics['understanding'] >= 0.999 and metrics['confidence'] >= 0.999
        }
