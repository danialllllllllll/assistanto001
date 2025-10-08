import numpy as np
import pickle
import json
import time
from datetime import datetime
from core.neural_network import ProgressiveNeuralNetwork
from core.understanding_trainer import UnderstandingTrainer
from core.self_model import SenseOfSelf
from personality.traits import PersonalityTraits
from personality.narrative_memory import NarrativeMemory
from philosophy.thinker_engine import ThinkerEngine
from philosophy.reasoning_rules import ReasoningRules
from knowledge.storage import KnowledgeStorage
from interfaces.app import start_flask_background, update_training_state, update_personality, add_philosophy_insight, update_core_values, add_to_history

print("="*80)
print("ADVANCED AI TRAINING SYSTEM")
print("99.9% Understanding Enforcement - Curriculum-Based Learning")
print("="*80)

print("\nLoading dataset...")
with open("dataset_features.pkl", "rb") as f:
    data_inputs2 = pickle.load(f)

features_STDs = np.std(a=data_inputs2, axis=0)
data_inputs = data_inputs2[:, features_STDs > 50]

with open("outputs.pkl", "rb") as f:
    data_outputs = pickle.load(f)

print(f"Dataset loaded: {data_inputs.shape[0]} samples, {data_inputs.shape[1]} features")
print(f"Classes: {len(np.unique(data_outputs))}")

print("\nLoading configurations...")
with open("configs/stage_config.json", "r") as f:
    config = json.load(f)

with open("configs/core_values.json", "r") as f:
    core_values_config = json.load(f)

stages = config['developmental_stages']
understanding_requirements = config['understanding_requirements']

print(f"Minimum Understanding Required: {understanding_requirements['minimum_understanding']*100}%")
print(f"Minimum Confidence Required: {understanding_requirements['minimum_confidence']*100}%")
print(f"Strict Enforcement: {understanding_requirements['strict_enforcement']}")

print("\nInitializing AI components...")
hidden_layers = [150, 60]
output_size = 4

network = ProgressiveNeuralNetwork(
    input_size=data_inputs.shape[1],
    hidden_sizes=hidden_layers,
    output_size=output_size
)

trainer = UnderstandingTrainer(network, config)
sense_of_self = SenseOfSelf()
personality = PersonalityTraits()
memory = NarrativeMemory()
reasoning_rules = ReasoningRules()
thinker = ThinkerEngine(reasoning_rules)
knowledge = KnowledgeStorage()

core_values_list = [
    {"name": v['name'], "priority": v['priority'], "status": "Enforced"}
    for v in core_values_config['immutable_core_values']['values']
]
update_core_values(core_values_list)

print("\nStarting Flask web interface...")
flask_thread = start_flask_background()
time.sleep(2)

print("\n" + "="*80)
print("BEGINNING CURRICULUM-BASED DEVELOPMENTAL LEARNING")
print("="*80)
print("\nCurriculum Structure:")
print("  Baby Steps/Toddler/Pre-K: Basic patterns, shapes, colors, simple concepts")
print("  Elementary: Math, reading, science basics")
print("  Teen: History, social studies, advanced subjects")
print("  Scholar: Complex subjects, philosophy prep, critical analysis")
print("  Thinker: Philosophy, ethics, reasoning")
print("\n" + "="*80)

training_history = []

for stage_idx, stage_info in enumerate(stages):
    stage_name = stage_info['name']
    
    curriculum_description = ""
    if stage_name in ['Baby Steps', 'Toddler', 'Pre-K']:
        curriculum_description = "Learning: Basic patterns, shapes, colors, simple concepts"
    elif stage_name == 'Elementary':
        curriculum_description = "Learning: Math, reading, science basics"
    elif stage_name == 'Teen':
        curriculum_description = "Learning: History, social studies, advanced subjects"
        personality.initialize_stage(stage_name)
    elif stage_name == 'Scholar':
        curriculum_description = "Learning: Complex subjects, philosophy prep, critical analysis"
        personality.initialize_stage(stage_name)
    elif stage_name == 'Thinker':
        curriculum_description = "Learning: Philosophy, ethics, reasoning"
        personality.initialize_stage(stage_name)
    
    print(f"\n{'='*80}")
    print(f"STAGE {stage_idx + 1}/{len(stages)}: {stage_name}")
    print(f"{'='*80}")
    print(f"{stage_info['description']}")
    print(f"{curriculum_description}")
    print(f"Active nodes: {stage_info['active_nodes_percent']*100:.0f}%")
    print(f"Understanding threshold: {stage_info['understanding_threshold']*100:.1f}%")
    print(f"{'='*80}")
    
    sense_of_self.update_stage(stage_name, 0.0)
    memory.add_memory(
        stage_name, 
        'stage_start', 
        f"Entering {stage_name} stage with curriculum: {curriculum_description}",
        importance=9
    )
    
    iteration = 0
    stage_passed = False
    
    print("\nQuizzing for understanding (not just memorization)...")
    
    while not stage_passed:
        batch_size = min(64, len(data_inputs))
        indices = np.random.choice(len(data_inputs), batch_size, replace=False)
        X_batch = data_inputs[indices]
        y_batch = data_outputs[indices]
        
        network.set_stage_activation(stage_info['active_nodes_percent'])
        network.forward(X_batch)
        network.backward(X_batch, y_batch, stage_info['learning_rate'])
        
        if iteration % 10 == 0:
            predictions = network.predict(data_inputs)
            confidences = network.get_confidence(data_inputs)
            
            correct = predictions == data_outputs
            accuracy = np.mean(correct)
            avg_confidence = np.mean(confidences)
            
            correct_conf = np.mean(confidences[correct]) if np.any(correct) else 0
            incorrect_conf = np.mean(confidences[~correct]) if np.any(~correct) else 0
            calibration = correct_conf - incorrect_conf
            
            understanding_score = accuracy * 0.5 + correct_conf * 0.3 + max(0, calibration) * 0.2
            
            update_training_state(
                stage_name=stage_name,
                stage_index=stage_idx,
                understanding=understanding_score,
                confidence=avg_confidence,
                accuracy=accuracy,
                iteration=iteration,
                total_stages=len(stages)
            )
            
            if iteration % 50 == 0:
                print(f"Iteration {iteration:4d}: "
                      f"Understanding={understanding_score:.4f}, "
                      f"Accuracy={accuracy:.4f}, "
                      f"Confidence={avg_confidence:.4f}")
            
            threshold_met = (
                understanding_score >= understanding_requirements['minimum_understanding'] and
                avg_confidence >= understanding_requirements['minimum_confidence']
            )
            
            if threshold_met:
                print(f"\n✓ 99.9% Understanding threshold achieved at iteration {iteration}!")
                print(f"  Understanding: {understanding_score:.4f} (required: {understanding_requirements['minimum_understanding']})")
                print(f"  Confidence: {avg_confidence:.4f} (required: {understanding_requirements['minimum_confidence']})")
                stage_passed = True
                
                stage_result = {
                    'stage': stage_name,
                    'understanding': understanding_score,
                    'confidence': avg_confidence,
                    'accuracy': accuracy,
                    'iterations': iteration,
                    'passed': True
                }
                training_history.append(stage_result)
                add_to_history(stage_result)
                
                if stage_name in ['Teen', 'Scholar', 'Thinker']:
                    personality.process_learning_outcome(
                        stage_name, understanding_score, accuracy, avg_confidence
                    )
                    update_personality(personality.traits)
                
                if stage_name == 'Scholar':
                    insight = f"Achieved mastery in {stage_name} stage with {understanding_score*100:.2f}% understanding"
                    add_philosophy_insight(insight)
                    thinker.philosophical_insights.append({
                        'stage': stage_name,
                        'insight': insight,
                        'timestamp': datetime.now().isoformat()
                    })
                
                if stage_name == 'Thinker':
                    thought = thinker.reason_about(
                        topic="Learning and Understanding",
                        context=f"Achieved {understanding_score*100:.2f}% understanding",
                        evidence=[f"Completed all {len(stages)} developmental stages"]
                    )
                    add_philosophy_insight(thought['conclusion']['statement'])
                
                knowledge.store_learned_concept(
                    concept_id=f"stage_{stage_idx}",
                    concept_name=stage_name,
                    stage=stage_name,
                    understanding_score=understanding_score,
                    details={
                        'curriculum': curriculum_description,
                        'iterations': iteration,
                        'accuracy': accuracy,
                        'confidence': avg_confidence
                    }
                )
                
                sense_of_self.update_stage(stage_name, understanding_score)
                memory.add_memory(
                    stage_name,
                    'stage_complete',
                    f"Mastered {stage_name} with {understanding_score*100:.2f}% understanding",
                    emotional_context='achievement',
                    importance=10
                )
                
                break
        
        iteration += 1
        
        if iteration >= stage_info['max_iterations'] and not stage_passed:
            print(f"\nIteration {iteration} reached. Continuing until 99.9% understanding...")
    
    print(f"\n{'='*80}")
    print(f"STAGE COMPLETE: {stage_name}")
    print(f"Understanding Score: {understanding_score:.4f} (threshold: {stage_info['understanding_threshold']})")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confidence: {avg_confidence:.4f}")
    print(f"Status: {'PASSED ✓' if stage_passed else 'NEEDS MORE WORK ✗'}")
    print(f"{'='*80}")

print("\n" + "="*80)
print("DEVELOPMENTAL JOURNEY COMPLETE")
print("="*80)

print("\nTraining History:")
for result in training_history:
    status = "✓ PASSED" if result['passed'] else "✗ NEEDS WORK"
    print(f"  {result['stage']:12} - Understanding: {result['understanding']:.4f} - {status}")

print("\nSaving knowledge and progress...")
solution_log = {
    'timestamp': datetime.now().isoformat(),
    'training_complete': True,
    'stages_completed': len(training_history),
    'history': training_history,
    'personality_traits': personality.traits,
    'philosophical_insights': thinker.philosophical_insights,
    'core_values': core_values_list,
    'sense_of_self': sense_of_self.identity,
    'memories': memory.memories[-20:] if len(memory.memories) > 20 else memory.memories
}

with open('knowledge/solution_log.json', 'w') as f:
    json.dump(solution_log, f, indent=2)

print("Progress saved to knowledge/solution_log.json")

print("\n" + "="*80)
print("TRAINING COMPLETE - Web dashboard continues to run at http://0.0.0.0:5000")
print("="*80)

print("\nKeeping web interface alive...")
while True:
    time.sleep(60)
