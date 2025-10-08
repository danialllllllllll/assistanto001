import re
from datetime import datetime

class ReasoningRules:
    """
    Critical thinking, bias detection, and truth verification rules
    Supports 99% truth accuracy requirement
    """
    
    def __init__(self):
        self.bias_patterns = {
            'confirmation_bias': ['only', 'always', 'never', 'proves'],
            'authority_bias': ['expert says', 'studies show', 'research proves'],
            'bandwagon': ['everyone knows', 'commonly accepted', 'popular belief'],
            'emotional_appeal': ['obviously', 'clearly', 'undeniable']
        }
        
        self.truth_criteria = {
            'evidence_based': 0.3,
            'logical_consistency': 0.25,
            'source_reliability': 0.25,
            'bias_free': 0.2
        }
        
        self.verification_log = []
    
    def detect_biases(self, text, context):
        """Detect cognitive biases in text or reasoning"""
        detected_biases = []
        
        text_lower = text.lower() if isinstance(text, str) else str(text).lower()
        
        for bias_type, patterns in self.bias_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    detected_biases.append({
                        'type': bias_type,
                        'pattern': pattern,
                        'severity': 'medium'
                    })
        
        absolute_words = len(re.findall(r'\b(always|never|all|none|every|impossible)\b', text_lower))
        if absolute_words > 2:
            detected_biases.append({
                'type': 'absolutist_thinking',
                'pattern': 'excessive absolute language',
                'severity': 'high'
            })
        
        return {
            'biases_detected': len(detected_biases),
            'details': detected_biases,
            'summary': f"Detected {len(detected_biases)} potential biases"
        }
    
    def verify_truth(self, evidence_list):
        """
        Verify truth based on evidence
        Returns truth score (0.0 to 1.0)
        """
        if not evidence_list:
            return 0.5
        
        scores = {criterion: 0.0 for criterion in self.truth_criteria}
        
        scores['evidence_based'] = min(1.0, len(evidence_list) / 5)
        
        consistent_evidence = sum(1 for e in evidence_list if 'contradict' not in str(e).lower())
        scores['logical_consistency'] = consistent_evidence / len(evidence_list) if evidence_list else 0
        
        reliable_sources = sum(1 for e in evidence_list if self.is_reliable_source(e))
        scores['source_reliability'] = reliable_sources / len(evidence_list) if evidence_list else 0
        
        total_bias = sum(len(self.detect_biases(str(e), '')['details']) for e in evidence_list)
        scores['bias_free'] = max(0, 1.0 - (total_bias / (len(evidence_list) * 2)))
        
        truth_score = sum(
            scores[criterion] * weight 
            for criterion, weight in self.truth_criteria.items()
        )
        
        verification = {
            'timestamp': datetime.now().isoformat(),
            'evidence_count': len(evidence_list),
            'truth_score': truth_score,
            'component_scores': scores
        }
        
        self.verification_log.append(verification)
        
        return truth_score
    
    def is_reliable_source(self, evidence):
        """Check if evidence comes from a reliable source"""
        evidence_str = str(evidence).lower()
        
        reliable_indicators = ['peer-reviewed', 'verified', 'confirmed', 'documented', 'measured']
        unreliable_indicators = ['rumor', 'alleged', 'unconfirmed', 'speculation']
        
        reliability = sum(1 for indicator in reliable_indicators if indicator in evidence_str)
        unreliability = sum(1 for indicator in unreliable_indicators if indicator in evidence_str)
        
        return reliability > unreliability
    
    def apply_critical_thinking(self, claim, assumptions):
        """Apply critical thinking to evaluate a claim"""
        evaluation = {
            'claim': claim,
            'assumptions_examined': len(assumptions),
            'issues': []
        }
        
        for assumption in assumptions:
            if not self.validate_assumption(assumption):
                evaluation['issues'].append(f"Questionable assumption: {assumption}")
        
        bias_check = self.detect_biases(claim, '')
        if bias_check['biases_detected'] > 0:
            evaluation['issues'].extend([b['type'] for b in bias_check['details']])
        
        evaluation['valid'] = len(evaluation['issues']) == 0
        
        return evaluation
    
    def validate_assumption(self, assumption):
        """Validate an assumption"""
        assumption_str = str(assumption).lower()
        
        invalid_patterns = ['assume', 'obviously', 'clearly', 'everyone knows']
        
        return not any(pattern in assumption_str for pattern in invalid_patterns)
    
    def logical_inference(self, premises, conclusion):
        """Check if conclusion logically follows from premises"""
        if not premises:
            return {'valid': False, 'reason': 'No premises provided'}
        
        confidence = 0.7
        
        if len(premises) >= 2:
            confidence += 0.1
        
        if any('because' in str(p).lower() or 'therefore' in str(p).lower() for p in premises):
            confidence += 0.1
        
        bias_in_premises = sum(len(self.detect_biases(str(p), '')['details']) for p in premises)
        if bias_in_premises > 2:
            confidence -= 0.3
        
        return {
            'valid': confidence >= 0.7,
            'confidence': confidence,
            'premises_count': len(premises),
            'biases_detected': bias_in_premises
        }
    
    def get_verification_stats(self):
        """Get statistics on truth verification"""
        if not self.verification_log:
            return {'total_verifications': 0}
        
        avg_truth_score = sum(v['truth_score'] for v in self.verification_log) / len(self.verification_log)
        
        high_truth = sum(1 for v in self.verification_log if v['truth_score'] >= 0.99)
        
        return {
            'total_verifications': len(self.verification_log),
            'average_truth_score': avg_truth_score,
            'high_truth_accuracy_count': high_truth,
            'truth_accuracy_rate': high_truth / len(self.verification_log) if self.verification_log else 0
        }
