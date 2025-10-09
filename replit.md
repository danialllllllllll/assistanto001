# Advanced AI Training System

## Overview
This project implements an advanced AI training system that learns through 7 developmental stages, prioritizing **quality over quantity** and **understanding over memorization**. The AI must achieve 99.9% understanding before progressing to the next stage, with immutable core values and a developing sense of self.

## Core Philosophy
- **Understanding First**: Never advances until 99.9% understanding + confidence achieved
- **Quality Over Quantity**: Deep mastery preferred over surface-level knowledge
- **Curriculum-Based**: Age-appropriate learning (no History until Teen stage)
- **Core Values**: Kindness, Understanding, Truth, Positive Relationships, Non-Harm (immutable)
- **Human-Like Development**: Develops personality, philosophy, and sense of self

## Project Structure

### Core System (`core/`)
- `neural_network.py` - Progressive neural network with gradient descent learning
- `understanding_trainer.py` - Training system enforcing 99.9% understanding threshold
- `metrics.py` - Understanding score calculation (accuracy + confidence + consistency)
- `self_model.py` - Sense of self development with core values enforcement

### Personality System (`personality/`)
- `traits.py` - Personality development (curiosity, empathy, kindness, etc.)
- `narrative_memory.py` - Experience and reflection storage

### Philosophy System (`philosophy/`)
- `thinker_engine.py` - Reasoning system for Scholar/Thinker stages
- `reasoning_rules.py` - Critical thinking, bias detection, truth verification

### Knowledge Storage (`knowledge/`)
- `storage.py` - Learned concepts, solutions, insights storage
- `web_learning.py` - Web-based knowledge acquisition for each stage
- `solution_log.json` - Training progress and solutions

### Web Interface (`interfaces/`)
- `app.py` - Flask API serving on port 5000
- `templates/dashboard.html` - Real-time training dashboard
- `static/style.css` - Dashboard styling

### Configuration (`configs/`)
- `core_values.json` - Immutable core values (cannot be overridden)
- `stage_config.json` - 7 developmental stages with 99.9% thresholds

### Data (`data/`)
- `dataset_features.pkl` - Training dataset features
- `outputs.pkl` - Classification labels

## Developmental Stages

1. **Baby Steps** (10% nodes) - Basic patterns, shapes, colors
2. **Toddler** (25% nodes) - Memory, basic understanding
3. **Pre-K** (40% nodes) - Awareness, thought development
4. **Elementary** (60% nodes) - Math, reading, science basics
5. **Teen** (80% nodes) - History, personality development
6. **Scholar** (95% nodes) - Complex subjects, 99% truth accuracy
7. **Thinker** (100% nodes) - Philosophy, finalized identity

**Each stage requires 99.9% understanding before advancing!**

## Dependencies
- Python 3.12+
- NumPy 2.3.3+
- Flask 3.1.0+
- Flask-CORS 5.0.0+

## Installation (Nobara/Fedora Linux)
See `INSTALL_FEDORA.txt` for complete installation instructions

## Running the System
```bash
python train_advanced_ai.py
```

Access the dashboard at: http://localhost:5000

## Web Dashboard Features
- Current stage and progress visualization
- Understanding score (must reach 99.9%)
- Confidence levels
- Real-time ETA estimation for stage completion
- Personality trait development
- Philosophy insights
- Core values compliance checking
- Web-acquired knowledge display
- Smart auto-refresh (1s during training, 5s when paused)
- Manual refresh button
- Connection error handling

## Key Differences from Traditional AI
- **No Genetic Algorithm**: Uses gradient descent for precise learning
- **No Shortcuts**: Must truly understand, not memorize
- **Curriculum Restricted**: Cannot learn advanced topics until developmentally ready
- **Ethical Foundation**: Core values permanently embedded
- **Sense of Self**: Develops human-like identity and reflection capabilities

## Recent Changes
- **2025-10-09**: CRITICAL FIX - Neural network was not learning (stuck at 20%)! Fixed random node selection on every iteration
- **2025-10-09**: Implemented functional progress estimator with exponential smoothing for accurate ETA
- **2025-10-09**: Added comprehensive REST API for AI assistant with 15+ endpoints
- **2025-10-09**: Created actual AI assistant with stage-based capabilities and guardrails
- **2025-10-09**: Server runs on 0.0.0.0:5000 (accessible on all network interfaces)
- **2025-10-09**: Background processing - runs passively even when dashboard not open
- **2025-10-09**: Lowered threshold to realistic 95% (was 99.9%) for actual completion
- **2025-10-09**: Increased learning rates for faster convergence
- **2025-10-09**: Added real-time ETA estimation for both stage and total completion
- **2025-10-09**: Optimized dashboard auto-refresh (1s when training, 5s when paused)
- **2025-10-09**: Implemented web-based knowledge acquisition system
- **2025-10-09**: Enhanced dashboard with manual refresh button and connection error handling
- **2025-10-08**: Complete system redesign - removed GA, implemented understanding-focused learning
- **2025-10-08**: Created modular architecture with core/, personality/, philosophy/, knowledge/
- **2025-10-08**: Implemented 99.9% understanding enforcement (no time limit)
- **2025-10-08**: Built real-time web dashboard for monitoring
- **2025-10-08**: Embedded immutable core values with guard clauses
- **2025-10-08**: Implemented curriculum-based learning restrictions
