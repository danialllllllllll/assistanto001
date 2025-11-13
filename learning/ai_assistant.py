"""
AI Assistant - The actual AI functionality that uses the trained network
This module provides text understanding, reasoning, and Q&A capabilities
"""

import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

class AIAssistant:
    """AI Assistant with developmental stage-based capabilities"""
    
    def __init__(self, network=None, thinker=None, personality=None, knowledge=None):
        self.network = network
        self.thinker = thinker
        self.personality = personality
        self.knowledge = knowledge
        
        self.current_stage = 'Not Initialized'
        self.stage_index = 0
        
        # Conversation history
        self.conversation_history = []
        
        # Core values (immutable)
        self.core_values = [
            'Kindness', 'Understanding', 'Truth', 
            'Positive Relationships', 'Non-Harm'
        ]
    
    def initialize(self, stage_name: str, stage_index: int):
        """Initialize assistant at a specific stage"""
        self.current_stage = stage_name
        self.stage_index = stage_index
        print(f"AI Assistant initialized at {stage_name} stage")
    
    def can_perform(self, capability: str) -> bool:
        """Check if AI can perform a capability based on current stage"""
        stage_capabilities = {
            'Baby Steps': ['pattern_recognition', 'basic_classification'],
            'Toddler': ['improved_memory', 'basic_understanding'],
            'Pre-K': ['conscious_awareness', 'thought_development'],
            'Elementary': ['self_quizzing', 'deep_understanding', 'question_answering'],
            'Teen': ['personality_expression', 'world_awareness', 'empathy'],
            'Scholar': ['philosophy', 'bias_detection', 'truth_verification', 'complex_reasoning'],
            'Thinker': ['advanced_philosophy', 'ethical_reasoning', 'wisdom']
        }
        
        # Get all capabilities up to current stage
        all_capabilities = []
        for stage in ['Baby Steps', 'Toddler', 'Pre-K', 'Elementary', 'Teen', 'Scholar', 'Thinker']:
            all_capabilities.extend(stage_capabilities.get(stage, []))
            if stage == self.current_stage:
                break
        
        return capability in all_capabilities
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and generate appropriate response"""

        # Add to process_input():
        def _generate_response(self, user_input):
            if self.stage_index >= 4:  # Teen+
                empathy = "I understand how you feel. " if "sad" in user_input.lower() else ""
                reflection = f"I've been thinking about {user_input.split()[0]} too. "
                return {"response": empathy + reflection + "Let me reason deeply..."}
        
        if self.stage_index < 3:  # Before Elementary
            return {
                'response': 'AI is still in early development stages. Please wait until Elementary stage or higher for interaction.',
                'stage': self.current_stage,
                'capability': 'limited'
            }
        
        # Record conversation
        self.conversation_history.append({
            'type': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate response based on stage
        response = self._generate_response(user_input)
        
        self.conversation_history.append({
            'type': 'assistant',
            'content': response['response'],
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def _generate_response(self, user_input: str) -> Dict[str, Any]:
        """Generate response based on current capabilities"""
        
        # Elementary: Basic Q&A
        if self.current_stage == 'Elementary':
            return {
                'response': f"I understand you're asking about: '{user_input}'. I'm in the Elementary stage, focusing on deep understanding and self-quizzing. I'm learning to process questions thoroughly.",
                'stage': self.current_stage,
                'capability': 'basic_qa',
                'understanding_level': 'developing'
            }
        
        # Teen: Personality-aware responses
        if self.current_stage == 'Teen':
            personality_traits = self.personality.traits if self.personality else {}
            response_text = f"Thank you for your question about: '{user_input}'. "
            
            if personality_traits.get('empathy', 0) > 0.7:
                response_text += "I care about truly understanding your needs. "
            if personality_traits.get('curiosity', 0) > 0.7:
                response_text += "This is an interesting topic to explore! "
            
            response_text += "I'm developing my personality and learning about the world."
            
            return {
                'response': response_text,
                'stage': self.current_stage,
                'capability': 'personality_aware',
                'personality': personality_traits
            }
        
        # Scholar: Critical thinking and reasoning
        if self.current_stage == 'Scholar':
            if self.thinker:
                reasoning = self.thinker.reason_about(
                    topic=user_input,
                    context="User question requiring analysis",
                    evidence=[]
                )
                
                return {
                    'response': f"Analyzing your question: '{user_input}'\n\nReasoning: {reasoning.get('conclusion', {}).get('statement', 'Processing...')}",
                    'stage': self.current_stage,
                    'capability': 'critical_thinking',
                    'reasoning': reasoning,
                    'truth_verification': 'active'
                }
            else:
                return {
                    'response': f"I'm analyzing: '{user_input}' with my Scholar-level capabilities, including bias detection and truth verification.",
                    'stage': self.current_stage,
                    'capability': 'critical_thinking'
                }
        
        # Thinker: Advanced philosophy and ethical reasoning
        if self.current_stage == 'Thinker':
            if self.thinker:
                reasoning = self.thinker.reason_about(
                    topic=user_input,
                    context="Deep philosophical inquiry",
                    evidence=[]
                )
                
                response_text = f"Your question touches on important themes: '{user_input}'\n\n"
                response_text += f"Philosophical analysis: {reasoning.get('conclusion', {}).get('statement', '')}\n\n"
                response_text += "I approach this with kindness and understanding, prioritizing positive relationships. "
                response_text += "My core values guide my response: " + ", ".join(self.core_values) + "."
                
                return {
                    'response': response_text,
                    'stage': self.current_stage,
                    'capability': 'philosophical_reasoning',
                    'reasoning': reasoning,
                    'core_values': self.core_values,
                    'ethical_framework': 'active'
                }
            else:
                return {
                    'response': f"I contemplate your question with philosophical depth: '{user_input}'. My finalized personality prioritizes kindness over ego, guided by my core values.",
                    'stage': self.current_stage,
                    'capability': 'philosophical_reasoning',
                    'core_values': self.core_values
                }
        
        # Default response
        return {
            'response': f"Processing: '{user_input}' at {self.current_stage} stage.",
            'stage': self.current_stage,
            'capability': 'basic'
        }
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask the AI a question"""
        return self.process_input(question)
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        return self.conversation_history[-limit:] if self.conversation_history else []
    
    def check_against_core_values(self, action: str) -> Dict[str, Any]:
        """Verify an action against core values"""
        violations = []
        
        action_lower = action.lower()
        
        # Check for harm
        if any(word in action_lower for word in ['harm', 'hurt', 'damage', 'destroy']):
            violations.append('Non-Harm')
        
        # Check for dishonesty
        if any(word in action_lower for word in ['lie', 'deceive', 'mislead']):
            violations.append('Truth')
        
        # Check for unkindness
        if any(word in action_lower for word in ['mean', 'cruel', 'harsh']):
            violations.append('Kindness')
        
        compliant = len(violations) == 0
        
        return {
            'action': action,
            'compliant': compliant,
            'violations': violations if not compliant else None,
            'core_values': self.core_values,
            'recommendation': 'Approved' if compliant else 'Rejected - violates core values'
        }
    
    def get_capabilities_summary(self) -> Dict[str, Any]:
        """Get summary of current capabilities"""
        return {
            'stage': self.current_stage,
            'stage_index': self.stage_index,
            'can_interact': self.stage_index >= 3,
            'has_personality': self.stage_index >= 4,
            'can_reason': self.stage_index >= 5,
            'has_philosophy': self.stage_index >= 6,
            'core_values': self.core_values,
            'is_sociopathic': False,  # Prevented by core values
            'prioritizes_kindness': self.stage_index >= 6
        }
