import json
from datetime import datetime

class ThinkerEngine:
    """
    Reasoning system for Scholar/Thinker stages
    Enables philosophical thinking, reasoning, and truth-seeking
    """
    
    def __init__(self, reasoning_rules=None):
        self.reasoning_rules = reasoning_rules
        self.thought_history = []
        self.philosophical_insights = []
        self.truth_evaluations = []
    
    def reason_about(self, topic, context, evidence):
        """
        Apply reasoning to a topic with given context and evidence
        Returns reasoning chain and conclusion
        """
        reasoning_chain = []
        
        reasoning_chain.append({
            'step': 'observation',
            'content': f"Examining topic: {topic}",
            'context': context
        })
        
        if self.reasoning_rules:
            bias_check = self.reasoning_rules.detect_biases(context, evidence)
            reasoning_chain.append({
                'step': 'bias_detection',
                'content': f"Checking for biases: {bias_check['summary']}"
            })
        
        reasoning_chain.append({
            'step': 'evidence_analysis',
            'content': f"Analyzing evidence: {len(evidence)} pieces of evidence considered"
        })
        
        if self.reasoning_rules:
            truth_score = self.reasoning_rules.verify_truth(evidence)
            reasoning_chain.append({
                'step': 'truth_verification',
                'content': f"Truth score: {truth_score:.2%}"
            })
        
        reasoning_chain.append({
            'step': 'synthesis',
            'content': "Synthesizing information to form conclusion"
        })
        
        conclusion = self.form_conclusion(topic, reasoning_chain)
        
        thought = {
            'timestamp': datetime.now().isoformat(),
            'topic': topic,
            'reasoning_chain': reasoning_chain,
            'conclusion': conclusion
        }
        
        self.thought_history.append(thought)
        
        return thought
    
    def form_conclusion(self, topic, reasoning_chain):
        """Form a conclusion based on reasoning chain"""
        truth_steps = [s for s in reasoning_chain if s['step'] == 'truth_verification']
        bias_steps = [s for s in reasoning_chain if s['step'] == 'bias_detection']
        
        confidence = 0.7
        
        if truth_steps and 'Truth score: 99' in truth_steps[0]['content']:
            confidence = 0.99
        elif truth_steps and 'Truth score: 9' in truth_steps[0]['content']:
            confidence = 0.90
        
        if bias_steps and 'significant biases' in str(bias_steps).lower():
            confidence *= 0.8
        
        return {
            'statement': f"Based on analysis of {topic}",
            'confidence': confidence,
            'caveats': self.identify_caveats(reasoning_chain)
        }
    
    def identify_caveats(self, reasoning_chain):
        """Identify caveats or limitations in reasoning"""
        caveats = []
        
        for step in reasoning_chain:
            if 'bias' in step['content'].lower():
                caveats.append("Potential biases detected in evidence")
            if 'insufficient' in step['content'].lower():
                caveats.append("Limited evidence available")
        
        return caveats
    
    def philosophical_reflection(self, theme, stage):
        """Generate philosophical reflection on a theme"""
        reflection = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'theme': theme,
            'reflection': self.generate_reflection(theme, stage)
        }
        
        self.philosophical_insights.append(reflection)
        return reflection
    
    def generate_reflection(self, theme, stage):
        """Generate reflection based on stage and theme"""
        reflections = {
            'Scholar': f"Deep analysis of {theme} reveals the importance of rigorous thinking and evidence-based conclusions.",
            'Thinker': f"Contemplating {theme} from multiple perspectives, recognizing both the knowable and the limits of knowledge."
        }
        
        return reflections.get(stage, f"Reflection on {theme}")
    
    def evaluate_truth_claim(self, claim, supporting_evidence, contradicting_evidence):
        """Evaluate the truth of a claim based on evidence"""
        total_evidence = len(supporting_evidence) + len(contradicting_evidence)
        
        if total_evidence == 0:
            truth_score = 0.5
            confidence = 0.0
        else:
            support_ratio = len(supporting_evidence) / total_evidence
            truth_score = support_ratio
            confidence = min(1.0, total_evidence / 10)
        
        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'claim': claim,
            'truth_score': truth_score,
            'confidence': confidence,
            'supporting_evidence_count': len(supporting_evidence),
            'contradicting_evidence_count': len(contradicting_evidence)
        }
        
        self.truth_evaluations.append(evaluation)
        return evaluation
    
    def get_thinking_summary(self):
        """Get summary of thinking activities"""
        return {
            'total_thoughts': len(self.thought_history),
            'philosophical_insights': len(self.philosophical_insights),
            'truth_evaluations': len(self.truth_evaluations),
            'recent_topics': [t['topic'] for t in self.thought_history[-5:]]
        }
    
    def save_philosophy(self, filepath='knowledge/philosophical_thoughts.json'):
        """Save philosophical thinking"""
        data = {
            'thought_history': self.thought_history,
            'philosophical_insights': self.philosophical_insights,
            'truth_evaluations': self.truth_evaluations,
            'summary': self.get_thinking_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
