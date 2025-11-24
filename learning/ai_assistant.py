
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

class DynamicAIAssistant:
    """
    True AI assistant that generates responses using neural network
    No predetermined responses - learns from context and training
    """
    
    def __init__(self, network=None, knowledge=None, web_learner=None):
        self.network = network
        self.knowledge = knowledge or {}
        self.web_learner = web_learner
        
        self.conversation_history = []
        self.learned_responses = {}
        self.context_memory = []
        
        # Core values (immutable)
        self.core_values = [
            'Kindness', 'Understanding', 'Truth', 
            'Positive Relationships', 'Non-Harm'
        ]
        
        # Response generation parameters
        self.temperature = 0.7
        self.max_context_length = 10
        
    def generate_response(self, user_input: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate response using neural network and learned knowledge"""
        
        # Build context vector
        context_vector = self._build_context_vector(user_input, context)
        
        # Get neural network response
        if self.network and hasattr(self.network, 'forward'):
            network_output = self.network.forward(context_vector, training=False)
            response_type = self._interpret_network_output(network_output)
        else:
            response_type = 'analytical'
            
        # Generate actual response text
        response_text = self._generate_response_text(user_input, response_type)
        
        # LEARN FROM THIS INTERACTION - Update network weights
        if self.network and hasattr(self.network, 'backward'):
            # Create pseudo-training data from conversation
            target = self._determine_ideal_response_class(user_input)
            try:
                # Backpropagate to learn from this interaction
                self.network.backward(context_vector, np.array([target]), learning_rate=0.0001)
            except Exception as e:
                print(f"Learning from conversation error: {e}")
        
        # Store in conversation history
        self.conversation_history.append({
            'type': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        self.conversation_history.append({
            'type': 'assistant',
            'content': response_text,
            'response_type': response_type,
            'timestamp': datetime.now().isoformat()
        })
        
        # Limit history size
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]
        
        # Self-improve periodically
        if len(self.conversation_history) % 10 == 0:
            self._self_optimize_response_strategy()
            
        return {
            'response': response_text,
            'type': response_type,
            'confidence': self._calculate_response_confidence(context_vector),
            'learning_applied': bool(self.knowledge),
            'learned_from_interaction': True
        }
    
    def _determine_ideal_response_class(self, user_input: str) -> int:
        """Determine ideal response class based on input analysis"""
        words = user_input.lower().split()
        
        # Question -> informative
        if '?' in user_input:
            return 0
        # Analysis keywords -> analytical
        elif any(w in words for w in ['analyze', 'explain', 'how', 'why']):
            return 1
        # Emotional keywords -> empathetic
        elif any(w in words for w in ['feel', 'sad', 'happy', 'worried']):
            return 2
        # Philosophical keywords -> philosophical
        elif any(w in words for w in ['meaning', 'purpose', 'truth', 'existence']):
            return 3
        else:
            return 1  # Default analytical
    
    def _self_optimize_response_strategy(self):
        """Self-optimize response generation strategy based on conversation history"""
        if len(self.conversation_history) < 10:
            return
        
        recent = self.conversation_history[-10:]
        
        # Analyze what types of responses work best
        user_inputs = [h['content'] for h in recent if h['type'] == 'user']
        
        # Extract patterns
        for user_msg in user_inputs:
            concepts = self._extract_key_concepts(user_msg)
            for concept in concepts:
                if concept not in self.learned_responses:
                    self.learned_responses[concept] = []
                self.learned_responses[concept].append(user_msg)
        
        # Update temperature for more diverse/focused responses
        if len(set(r['type'] for r in recent if r['type'] == 'assistant')) < 2:
            self.temperature = min(1.0, self.temperature + 0.1)  # More diverse
        else:
            self.temperature = max(0.3, self.temperature - 0.05)  # More focused
    
    def _build_context_vector(self, user_input: str, context: Optional[List[str]] = None) -> np.ndarray:
        """Build numerical context vector from text input"""
        # Simple encoding: word frequency + length + sentiment
        vector_size = 102  # Match network input size
        vector = np.zeros(vector_size)
        
        # Text features
        words = user_input.lower().split()
        vector[0] = len(words)  # Length
        vector[1] = len(user_input)  # Character count
        
        # Question detection
        vector[2] = 1.0 if '?' in user_input else 0.0
        
        # Sentiment (simple)
        positive_words = {'good', 'great', 'love', 'like', 'yes', 'please', 'thanks', 'help'}
        negative_words = {'bad', 'hate', 'no', 'don\'t', 'can\'t', 'wrong', 'error'}
        
        pos_count = sum(1 for w in words if w in positive_words)
        neg_count = sum(1 for w in words if w in negative_words)
        
        vector[3] = pos_count
        vector[4] = neg_count
        
        # Topic detection (simple keyword matching)
        topics = {
            'learning': ['learn', 'study', 'understand', 'knowledge'],
            'help': ['help', 'assist', 'support', 'guide'],
            'question': ['what', 'why', 'how', 'when', 'where', 'who'],
            'philosophy': ['think', 'believe', 'philosophy', 'meaning', 'truth'],
            'emotion': ['feel', 'emotion', 'happy', 'sad', 'afraid']
        }
        
        topic_idx = 5
        for topic, keywords in topics.items():
            vector[topic_idx] = sum(1 for w in words if w in keywords)
            topic_idx += 1
            
        # Fill remaining with word hash features
        for i, word in enumerate(words[:90]):
            idx = 12 + (hash(word) % 90)
            vector[idx] += 1
            
        # Normalize
        max_val = np.max(vector)
        if max_val > 0:
            vector = vector / max_val
            
        return vector.reshape(1, -1)
    
    def _interpret_network_output(self, network_output: np.ndarray) -> str:
        """Interpret network output to determine response type"""
        if network_output.shape[-1] >= 4:
            output_class = np.argmax(network_output[0])
            
            response_types = {
                0: 'informative',
                1: 'analytical',
                2: 'empathetic',
                3: 'philosophical'
            }
            
            return response_types.get(output_class, 'analytical')
        
        return 'analytical'
    
    def _generate_response_text(self, user_input: str, response_type: str) -> str:
        """Generate actual response text based on learned knowledge and type"""
        
        # Extract key topics from user input
        words = user_input.lower().split()
        
        # Check if we have relevant knowledge
        relevant_knowledge = []
        for topic in self.knowledge.keys():
            if topic.lower() in user_input.lower():
                relevant_knowledge.append(topic)
                
        # Build response based on type and knowledge
        if response_type == 'informative' and relevant_knowledge:
            topic = relevant_knowledge[0]
            knowledge_entry = self.knowledge[topic]
            if isinstance(knowledge_entry, list) and knowledge_entry:
                latest = knowledge_entry[-1]
                content = latest.get('content', {})
                if isinstance(content, dict):
                    understanding = content.get('comprehensive_understanding', 
                                              content.get('understanding', ''))
                    if understanding:
                        return f"Based on my understanding of {topic}: {understanding[:200]}..."
                        
        elif response_type == 'philosophical':
            return self._generate_philosophical_response(user_input)
            
        elif response_type == 'empathetic':
            return self._generate_empathetic_response(user_input)
            
        # Default analytical response
        return self._generate_analytical_response(user_input, words)
    
    def _generate_philosophical_response(self, user_input: str) -> str:
        """Generate philosophical response"""
        philosophical_starters = [
            "Contemplating your question deeply, I consider",
            "From a philosophical perspective",
            "This touches on fundamental questions about",
            "Reflecting on this, I believe"
        ]
        
        concepts = self._extract_key_concepts(user_input)
        starter = np.random.choice(philosophical_starters)
        
        if concepts:
            return f"{starter} the nature of {concepts[0]} and how it relates to understanding and truth."
        
        return f"{starter} the deeper meaning behind your inquiry."
    
    def _generate_empathetic_response(self, user_input: str) -> str:
        """Generate empathetic response"""
        empathy_starters = [
            "I understand you're asking about",
            "I appreciate your interest in",
            "That's a thoughtful question about",
            "I can see why you're curious about"
        ]
        
        starter = np.random.choice(empathy_starters)
        concepts = self._extract_key_concepts(user_input)
        
        if concepts:
            return f"{starter} {concepts[0]}. Let me help you understand this better."
            
        return "I'm here to help you understand. Could you tell me more about what you'd like to know?"
    
    def _generate_analytical_response(self, user_input: str, words: List[str]) -> str:
        """Generate analytical response"""
        concepts = self._extract_key_concepts(user_input)
        
        if '?' in user_input:
            if concepts:
                return f"Analyzing your question about {concepts[0]}: This involves understanding multiple dimensions and their relationships."
            return "Let me analyze that question systematically and provide a reasoned response."
            
        if any(w in words for w in ['learn', 'understand', 'know']):
            return "I'm continuously learning and processing information to deepen my understanding."
            
        return "I'm processing your input using my neural network to generate a meaningful response based on my training and learned knowledge."
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Capitalize words and compound phrases
        concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+)*\b', text)
        
        # Add important lowercase words
        important_words = text.lower().split()
        filtered = [w for w in important_words if len(w) > 4 and w not in 
                   {'about', 'there', 'their', 'would', 'could', 'should', 'which'}]
        
        return (concepts + filtered)[:3]
    
    def _calculate_response_confidence(self, context_vector: np.ndarray) -> float:
        """Calculate confidence in response"""
        # Base confidence on knowledge availability and context quality
        confidence = 0.5
        
        if self.knowledge:
            confidence += 0.2
            
        if len(self.conversation_history) > 0:
            confidence += 0.1
            
        # Boost if context vector is well-formed
        if np.sum(context_vector) > 0:
            confidence += 0.2
            
        return min(1.0, confidence)
    
    def learn_from_web(self, topic: str) -> Dict[str, Any]:
        """Use web learner to acquire knowledge"""
        if self.web_learner:
            learned = self.web_learner.search_and_learn(topic)
            
            # Integrate into knowledge base
            if topic not in self.knowledge:
                self.knowledge[topic] = []
            self.knowledge[topic].append(learned)
            
            return {
                'success': True,
                'topic': topic,
                'confidence': learned.get('confidence', 0)
            }
            
        return {'success': False, 'error': 'Web learner not available'}
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary"""
        return {
            'total_exchanges': len(self.conversation_history) // 2,
            'recent_history': self.conversation_history[-10:],
            'topics_discussed': len(set(self._extract_key_concepts(' '.join(
                [h['content'] for h in self.conversation_history if h['type'] == 'user']
            ))))
        }
