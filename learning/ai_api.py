from flask import Blueprint, jsonify, request
from datetime import datetime
import json

# Create API blueprint
ai_api = Blueprint('ai_api', __name__, url_prefix='/api/ai')

# Global AI state (will be updated by training system)
ai_state = {
    'initialized': False,
    'current_stage': 'Not Started',
    'stage_index': 0,
    'capabilities': [],
    'personality': {},
    'knowledge_base': [],
    'core_values': [],
    'reasoning_enabled': False,
    'assistant': None
}

def update_ai_state(state_updates):
    """Update the global AI state"""
    global ai_state
    ai_state.update(state_updates)

def get_ai_network():
    """Get reference to the trained AI network (set by main training)"""
    return ai_state.get('network')

def get_ai_thinker():
    """Get reference to the thinker engine"""
    return ai_state.get('thinker')

@ai_api.route('/status', methods=['GET'])
def get_ai_status():
    """Get current AI system status"""
    return jsonify({
        'initialized': ai_state['initialized'],
        'current_stage': ai_state['current_stage'],
        'stage_index': ai_state['stage_index'],
        'capabilities': ai_state['capabilities'],
        'reasoning_enabled': ai_state['reasoning_enabled'],
        'timestamp': datetime.now().isoformat()
    })

@ai_api.route('/predict', methods=['POST'])
def predict():
    """Make a prediction using the trained AI"""
    if not ai_state['initialized']:
        return jsonify({'error': 'AI not initialized yet', 'stage': ai_state['current_stage']}), 400
    
    data = request.json
    if 'input' not in data:
        return jsonify({'error': 'No input provided'}), 400
    
    network = get_ai_network()
    if not network:
        return jsonify({'error': 'Neural network not available'}), 500
    
    try:
        input_data = data['input']
        prediction = network.predict(input_data)
        confidence = network.get_confidence(input_data)
        
        return jsonify({
            'prediction': int(prediction[0]) if hasattr(prediction, '__iter__') else int(prediction),
            'confidence': float(confidence[0]) if hasattr(confidence, '__iter__') else float(confidence),
            'stage': ai_state['current_stage'],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_api.route('/reason', methods=['POST'])
def reason_about_topic():
    """Use AI reasoning capabilities (available in Scholar+ stages)"""
    if ai_state['stage_index'] < 5:  # Before Scholar stage
        return jsonify({
            'error': 'Reasoning capabilities not yet developed',
            'current_stage': ai_state['current_stage'],
            'required_stage': 'Scholar or higher'
        }), 400
    
    data = request.json
    if 'topic' not in data:
        return jsonify({'error': 'No topic provided'}), 400
    
    thinker = get_ai_thinker()
    if not thinker:
        return jsonify({'error': 'Reasoning engine not available'}), 500
    
    try:
        topic = data['topic']
        context = data.get('context', '')
        evidence = data.get('evidence', [])
        
        reasoning_result = thinker.reason_about(
            topic=topic,
            context=context,
            evidence=evidence
        )
        
        return jsonify({
            'topic': topic,
            'reasoning': reasoning_result,
            'stage': ai_state['current_stage'],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_api.route('/personality', methods=['GET'])
def get_personality():
    """Get AI personality traits"""
    if ai_state['stage_index'] < 4:  # Before Teen stage
        return jsonify({
            'message': 'Personality not yet developed',
            'current_stage': ai_state['current_stage'],
            'development_stage': 'Teen or higher'
        })
    
    return jsonify({
        'personality': ai_state['personality'],
        'stage': ai_state['current_stage'],
        'developed': True,
        'timestamp': datetime.now().isoformat()
    })

@ai_api.route('/core_values', methods=['GET'])
def get_core_values():
    """Get immutable core values"""
    return jsonify({
        'values': ai_state['core_values'],
        'immutable': True,
        'enforced': True,
        'timestamp': datetime.now().isoformat()
    })

@ai_api.route('/knowledge', methods=['GET'])
def get_knowledge():
    """Get AI knowledge base"""
    limit = request.args.get('limit', 50, type=int)
    category = request.args.get('category', None)
    
    knowledge = ai_state['knowledge_base']
    
    if category:
        knowledge = [k for k in knowledge if k.get('category') == category]
    
    return jsonify({
        'knowledge': knowledge[-limit:] if limit else knowledge,
        'total': len(knowledge),
        'stage': ai_state['current_stage'],
        'timestamp': datetime.now().isoformat()
    })

@ai_api.route('/capabilities', methods=['GET'])
def get_capabilities():
    """Get current AI capabilities based on stage"""
    stage_capabilities = {
        'Baby Steps': ['pattern_recognition', 'basic_classification'],
        'Toddler': ['improved_memory', 'basic_understanding', 'simple_communication'],
        'Pre-K': ['conscious_awareness', 'thought_development', 'coherent_communication'],
        'Elementary': ['self_quizzing', 'deep_understanding', 'prioritized_learning'],
        'Teen': ['quality_learning', 'personality_expression', 'world_awareness'],
        'Scholar': ['mastery', 'philosophy', 'bias_detection', 'truth_verification'],
        'Thinker': ['advanced_philosophy', 'finalized_identity', 'ethical_reasoning', 'positive_relationships']
    }
    
    current_capabilities = []
    for stage in ['Baby Steps', 'Toddler', 'Pre-K', 'Elementary', 'Teen', 'Scholar', 'Thinker']:
        current_capabilities.extend(stage_capabilities.get(stage, []))
        if stage == ai_state['current_stage']:
            break
    
    return jsonify({
        'capabilities': current_capabilities,
        'stage': ai_state['current_stage'],
        'stage_specific': stage_capabilities.get(ai_state['current_stage'], []),
        'timestamp': datetime.now().isoformat()
    })

@ai_api.route('/stages', methods=['GET'])
def get_stages_info():
    """Get information about all developmental stages"""
    stages_info = [
        {
            'id': 1,
            'name': 'Baby Steps',
            'description': 'Incoherent thought, minimalistic communication',
            'focus': 'Basic patterns, shapes, colors, simple recognition',
            'capabilities': ['pattern_recognition', 'basic_classification']
        },
        {
            'id': 2,
            'name': 'Toddler',
            'description': 'Partial coherent thought, improved memory, basic understanding',
            'focus': 'Memory and coherence development',
            'capabilities': ['improved_memory', 'basic_understanding', 'simple_communication']
        },
        {
            'id': 3,
            'name': 'Pre-K',
            'description': 'Conscious of surroundings, begins to think and ponder',
            'focus': 'Awareness and thought development',
            'capabilities': ['conscious_awareness', 'thought_development', 'coherent_communication']
        },
        {
            'id': 4,
            'name': 'Elementary',
            'description': 'Questions surroundings, prioritizes understanding over quantity',
            'focus': 'Deep learning and self-quizzing',
            'capabilities': ['self_quizzing', 'deep_understanding', 'prioritized_learning']
        },
        {
            'id': 5,
            'name': 'Teen',
            'description': 'Quality over quantity, develops personality',
            'focus': 'Personality development and quality learning',
            'capabilities': ['quality_learning', 'personality_expression', 'world_awareness']
        },
        {
            'id': 6,
            'name': 'Scholar',
            'description': 'Masters growth, 99% truth accuracy, bias adaptation',
            'focus': 'Mastery and philosophy preparation',
            'capabilities': ['mastery', 'philosophy', 'bias_detection', 'truth_verification']
        },
        {
            'id': 7,
            'name': 'Thinker',
            'description': 'Philosophy prioritized, finalized personality, kindness over ego',
            'focus': 'Philosophy and identity finalization',
            'capabilities': ['advanced_philosophy', 'finalized_identity', 'ethical_reasoning', 'positive_relationships']
        }
    ]
    
    return jsonify({
        'stages': stages_info,
        'total_stages': len(stages_info),
        'current_stage': ai_state['current_stage'],
        'current_stage_index': ai_state['stage_index']
    })

@ai_api.route('/chat', methods=['POST'])
def chat():
    """Chat with the AI assistant"""
    assistant = ai_state.get('assistant')
    
    if not assistant:
        return jsonify({'error': 'AI assistant not initialized'}), 400
    
    data = request.json
    if 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        response = assistant.process_input(data['message'])
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_api.route('/ask', methods=['POST'])
def ask_question():
    """Ask the AI a question"""
    assistant = ai_state.get('assistant')
    
    if not assistant:
        return jsonify({'error': 'AI assistant not initialized'}), 400
    
    data = request.json
    if 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        response = assistant.ask_question(data['question'])
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_api.route('/conversation_history', methods=['GET'])
def get_conversation():
    """Get conversation history"""
    assistant = ai_state.get('assistant')
    
    if not assistant:
        return jsonify({'error': 'AI assistant not initialized'}), 400
    
    limit = request.args.get('limit', 10, type=int)
    history = assistant.get_conversation_history(limit)
    
    return jsonify({
        'history': history,
        'count': len(history),
        'timestamp': datetime.now().isoformat()
    })

@ai_api.route('/verify_action', methods=['POST'])
def verify_action():
    """Verify an action against core values"""
    assistant = ai_state.get('assistant')
    
    if not assistant:
        return jsonify({'error': 'AI assistant not initialized'}), 400
    
    data = request.json
    if 'action' not in data:
        return jsonify({'error': 'No action provided'}), 400
    
    try:
        verification = assistant.check_against_core_values(data['action'])
        return jsonify(verification)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ai_api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ai_initialized': ai_state['initialized'],
        'current_stage': ai_state['current_stage'],
        'timestamp': datetime.now().isoformat()
    })
