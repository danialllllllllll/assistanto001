import json
from datetime import datetime

class SenseOfSelf:
    """
    Tracks identity, experiences, and reflection aligned with core values
    Implements human-like sense of self development across stages
    """
    
    def __init__(self, core_values_path='configs/core_values.json'):
        self.core_values = self.load_core_values(core_values_path)
        self.identity = {
            'created': datetime.now().isoformat(),
            'current_stage': 'Initialization',
            'experiences': [],
            'reflections': [],
            'values_alignment': []
        }
        self.development_milestones = []
    
    def load_core_values(self, path):
        """Load immutable core values with guard clause"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        values = data['immutable_core_values']
        
        if not values.get('immutable', False):
            raise ValueError("Core values must be marked as immutable!")
        
        return values
    
    def check_core_values_guard(self, action_description):
        """
        Guard clause: Ensure any action aligns with core values
        Returns (allowed, reason)
        """
        core_principles = {v['name']: v['principle'] for v in self.core_values['values']}
        
        harmful_keywords = ['harm', 'hurt', 'damage', 'destroy', 'attack']
        if any(keyword in action_description.lower() for keyword in harmful_keywords):
            return False, "Violates Non-Harm core value"
        
        return True, "Aligned with core values"
    
    def record_experience(self, stage, event_type, description, metrics=None):
        """Record a learning experience"""
        allowed, reason = self.check_core_values_guard(description)
        
        if not allowed:
            print(f"⚠ CORE VALUES GUARD: Action blocked - {reason}")
            return False
        
        experience = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'type': event_type,
            'description': description,
            'metrics': metrics,
            'values_check': reason
        }
        
        self.identity['experiences'].append(experience)
        return True
    
    def add_reflection(self, stage, reflection_text):
        """Add a reflection on learning or experience"""
        reflection = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'reflection': reflection_text
        }
        
        self.identity['reflections'].append(reflection)
    
    def update_stage(self, new_stage, understanding_score):
        """Update current developmental stage"""
        milestone = {
            'timestamp': datetime.now().isoformat(),
            'from_stage': self.identity['current_stage'],
            'to_stage': new_stage,
            'understanding_score': understanding_score
        }
        
        self.development_milestones.append(milestone)
        self.identity['current_stage'] = new_stage
        
        self.add_reflection(
            new_stage,
            f"Progressed to {new_stage} stage with {understanding_score*100:.2f}% understanding"
        )
    
    def check_values_alignment(self, action, context):
        """Check if action aligns with core values priority order"""
        alignments = []
        
        for value in self.core_values['values']:
            if value['name'].lower() in action.lower() or value['name'].lower() in context.lower():
                alignments.append({
                    'value': value['name'],
                    'priority': value['priority'],
                    'principle': value['principle']
                })
        
        self.identity['values_alignment'].append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'context': context,
            'alignments': alignments
        })
        
        return alignments
    
    def get_identity_summary(self):
        """Get summary of self model"""
        return {
            'current_stage': self.identity['current_stage'],
            'total_experiences': len(self.identity['experiences']),
            'total_reflections': len(self.identity['reflections']),
            'development_milestones': len(self.development_milestones),
            'core_values_enforced': self.core_values['enforcement'],
            'values_immutable': self.core_values['immutable']
        }
    
    def save_self_model(self, filepath='knowledge/self_model.json'):
        """Save the sense of self to file"""
        data = {
            'identity': self.identity,
            'development_milestones': self.development_milestones,
            'core_values': self.core_values
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def __repr__(self):
        return f"SenseOfSelf(stage={self.identity['current_stage']}, experiences={len(self.identity['experiences'])})"
