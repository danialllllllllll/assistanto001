import json
from datetime import datetime

class PersonalityTraits:
    """
    Track personality development through stages (Teen onwards)
    Personality evolves based on learning interactions, aligned with core values
    """
    
    def __init__(self):
        self.traits = {
            'curiosity': 0.5,
            'empathy': 0.5,
            'critical_thinking': 0.5,
            'patience': 0.5,
            'kindness': 1.0,
            'humility': 0.5,
            'confidence': 0.5
        }
        
        self.trait_history = []
        self.stage_personalities = {}
        self.current_stage = None
        
        self.development_thresholds = {
            'Teen': 0.6,
            'Scholar': 0.8,
            'Thinker': 0.95
        }
    
    def initialize_stage(self, stage_name):
        """Initialize personality for a new stage"""
        self.current_stage = stage_name
        
        if stage_name not in self.stage_personalities:
            self.stage_personalities[stage_name] = {
                'started': datetime.now().isoformat(),
                'traits_snapshot': self.traits.copy(),
                'interactions': [],
                'growth': []
            }
    
    def update_trait(self, trait_name, change, reason):
        """Update a personality trait based on experience"""
        if trait_name == 'kindness':
            print("⚠ Kindness is a core value and cannot be decreased")
            return
        
        if trait_name in self.traits:
            old_value = self.traits[trait_name]
            self.traits[trait_name] = max(0.0, min(1.0, self.traits[trait_name] + change))
            
            change_record = {
                'timestamp': datetime.now().isoformat(),
                'stage': self.current_stage,
                'trait': trait_name,
                'old_value': old_value,
                'new_value': self.traits[trait_name],
                'change': change,
                'reason': reason
            }
            
            self.trait_history.append(change_record)
            
            if self.current_stage in self.stage_personalities:
                self.stage_personalities[self.current_stage]['growth'].append(change_record)
    
    def process_learning_outcome(self, stage_name, understanding, accuracy, confidence):
        """Evolve personality based on learning outcomes"""
        self.initialize_stage(stage_name)
        
        if understanding >= 0.999:
            self.update_trait('confidence', 0.05, 'Achieved high understanding')
            self.update_trait('critical_thinking', 0.03, 'Deep understanding demonstrates critical thinking')
        
        if confidence >= 0.999:
            self.update_trait('confidence', 0.03, 'High confidence in correct answers')
        elif confidence < 0.5:
            self.update_trait('humility', 0.02, 'Recognizing uncertainty')
        
        if accuracy >= 0.95:
            self.update_trait('curiosity', 0.02, 'Successful learning breeds curiosity')
        
        if stage_name in ['Scholar', 'Thinker']:
            self.update_trait('critical_thinking', 0.04, f'{stage_name} stage development')
            self.update_trait('patience', 0.02, f'{stage_name} stage requires patience')
    
    def process_interaction(self, interaction_type, context, outcome):
        """Process an interaction and update personality accordingly"""
        if self.current_stage not in self.stage_personalities:
            self.initialize_stage(self.current_stage or 'Unknown')
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'type': interaction_type,
            'context': context,
            'outcome': outcome
        }
        
        self.stage_personalities[self.current_stage]['interactions'].append(interaction)
        
        if 'help' in context.lower() or 'assist' in context.lower():
            self.update_trait('empathy', 0.02, 'Helpful interaction')
            self.update_trait('kindness', 0.01, 'Kind assistance')
        
        if 'question' in context.lower() or 'why' in context.lower():
            self.update_trait('curiosity', 0.02, 'Questioning interaction')
        
        if 'mistake' in outcome.lower() or 'error' in outcome.lower():
            self.update_trait('humility', 0.03, 'Learning from mistakes')
    
    def get_dominant_traits(self, n=3):
        """Get the n most dominant personality traits"""
        sorted_traits = sorted(self.traits.items(), key=lambda x: x[1], reverse=True)
        return sorted_traits[:n]
    
    def get_personality_profile(self):
        """Get current personality profile"""
        dominant = self.get_dominant_traits(3)
        
        return {
            'current_stage': self.current_stage,
            'all_traits': self.traits.copy(),
            'dominant_traits': {trait: value for trait, value in dominant},
            'total_interactions': sum(
                len(stage_data['interactions']) 
                for stage_data in self.stage_personalities.values()
            ),
            'growth_events': len(self.trait_history)
        }
    
    def finalize_personality(self):
        """Finalize personality at Thinker stage"""
        if self.current_stage == 'Thinker':
            return {
                'finalized': True,
                'final_traits': self.traits.copy(),
                'personality_summary': self.generate_personality_summary(),
                'timestamp': datetime.now().isoformat()
            }
        return None
    
    def generate_personality_summary(self):
        """Generate a summary description of the personality"""
        dominant = self.get_dominant_traits(3)
        
        summary = f"An AI characterized by "
        traits_desc = []
        
        for trait, value in dominant:
            if value >= 0.8:
                traits_desc.append(f"strong {trait}")
            elif value >= 0.6:
                traits_desc.append(f"notable {trait}")
            else:
                traits_desc.append(f"developing {trait}")
        
        summary += ", ".join(traits_desc)
        summary += ". "
        
        if self.traits['kindness'] >= 0.9:
            summary += "Prioritizes kindness in all interactions. "
        
        if self.traits['critical_thinking'] >= 0.8:
            summary += "Demonstrates advanced critical thinking abilities. "
        
        return summary
    
    def save_personality(self, filepath='knowledge/personality_profile.json'):
        """Save personality development"""
        data = {
            'current_traits': self.traits,
            'trait_history': self.trait_history,
            'stage_personalities': self.stage_personalities,
            'personality_profile': self.get_personality_profile()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
