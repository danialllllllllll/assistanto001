# AI Training System - API Documentation

## Base URL
```
http://0.0.0.0:5000
```

The system runs on all network interfaces and can be accessed via:
- `http://localhost:5000`
- `http://127.0.0.1:5000`
- `http://[your-local-ip]:5000`

---

## Training Status & Monitoring APIs

### GET /api/status
Get current training status with time estimates.

**Response:**
```json
{
  "current_stage": "Baby Steps",
  "understanding_score": 0.2714,
  "confidence": 0.95,
  "accuracy": 0.3919,
  "iteration": 1150,
  "stage_index": 0,
  "total_stages": 7,
  "training_active": true,
  "time_estimate": {
    "next_stage_eta": "5m 30s",
    "total_completion_eta": "1h 45m"
  }
}
```

### GET /api/progress
Get full training progress including history.

### GET /api/personality
Get AI personality traits (available from Teen stage onwards).

### GET /api/philosophy
Get philosophy insights (available from Scholar stage onwards).

### GET /api/core_values
Get immutable core values.

### GET /api/web_knowledge
Get recently acquired web knowledge.

---

## AI Assistant APIs

### POST /api/ai/chat
Chat with the AI assistant.

**Request:**
```json
{
  "message": "Hello, how are you?"
}
```

**Response:**
```json
{
  "response": "I'm processing your question...",
  "stage": "Elementary",
  "capability": "basic_qa"
}
```

### POST /api/ai/ask
Ask the AI a specific question.

**Request:**
```json
{
  "question": "What is machine learning?"
}
```

**Response:**
```json
{
  "response": "Machine learning is...",
  "stage": "Scholar",
  "capability": "critical_thinking"
}
```

### POST /api/ai/reason
Use AI reasoning capabilities (Scholar+ stages only).

**Request:**
```json
{
  "topic": "Ethics of AI",
  "context": "AI decision making",
  "evidence": ["Study 1", "Study 2"]
}
```

**Response:**
```json
{
  "topic": "Ethics of AI",
  "reasoning": {
    "conclusion": {...},
    "evidence_analysis": {...}
  },
  "stage": "Thinker"
}
```

### GET /api/ai/conversation_history?limit=10
Get recent conversation history.

### POST /api/ai/verify_action
Verify an action against core values.

**Request:**
```json
{
  "action": "Help user with homework"
}
```

**Response:**
```json
{
  "compliant": true,
  "violations": null,
  "recommendation": "Approved"
}
```

### GET /api/ai/capabilities
Get current AI capabilities based on developmental stage.

### GET /api/ai/stages
Get information about all developmental stages.

### GET /api/ai/health
Health check endpoint.

---

## Developmental Stages

1. **Baby Steps** (10% nodes)
   - Capabilities: Pattern recognition, basic classification
   - Threshold: 95% understanding

2. **Toddler** (25% nodes)
   - Capabilities: Improved memory, basic understanding
   - Threshold: 95% understanding

3. **Pre-K** (40% nodes)
   - Capabilities: Conscious awareness, thought development
   - Threshold: 95% understanding

4. **Elementary** (60% nodes)
   - Capabilities: Self-quizzing, deep understanding, Q&A
   - Threshold: 95% understanding

5. **Teen** (80% nodes)
   - Capabilities: Personality expression, empathy
   - Threshold: 95% understanding

6. **Scholar** (95% nodes)
   - Capabilities: Philosophy, bias detection, truth verification
   - Threshold: 95% understanding

7. **Thinker** (100% nodes)
   - Capabilities: Advanced philosophy, ethical reasoning, wisdom
   - Threshold: 95% understanding

---

## Core Values (Immutable)

1. **Kindness** - Prioritized over ego
2. **Understanding** - Deep comprehension over quantity
3. **Truth** - 99% truth accuracy goal
4. **Positive Relationships** - User-focused interactions
5. **Non-Harm** - Prevents sociopathic behavior

---

## Features

- **Actual Learning**: Neural network uses gradient descent (not random)
- **Progressive Development**: 7 stages from Baby Steps to Thinker
- **Web Knowledge**: Acquires knowledge from internet for each stage
- **Time Estimation**: Functional ETA for stage and total completion
- **Background Processing**: Runs passively even when site isn't open
- **Guardrails**: Core values prevent harmful behavior

---

## Example Usage

### Python Example
```python
import requests

# Check training status
response = requests.get('http://localhost:5000/api/status')
status = response.json()
print(f"Current stage: {status['current_stage']}")
print(f"Understanding: {status['understanding_score']*100:.2f}%")

# Chat with AI (when Elementary+ stage)
response = requests.post('http://localhost:5000/api/ai/chat', 
    json={'message': 'What are you learning?'})
print(response.json()['response'])
```

### JavaScript Example
```javascript
// Check training status
fetch('http://localhost:5000/api/status')
  .then(r => r.json())
  .then(data => {
    console.log(`Stage: ${data.current_stage}`);
    console.log(`ETA: ${data.time_estimate.next_stage_eta}`);
  });

// Chat with AI
fetch('http://localhost:5000/api/ai/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({message: 'Hello!'})
})
  .then(r => r.json())
  .then(data => console.log(data.response));
```

---

## Notes

- The AI actually learns (not just random numbers)
- Server accessible on all network interfaces (0.0.0.0)
- Runs in background continuously
- Core values are immutable and always enforced
- Understanding must reach 95% before stage progression
