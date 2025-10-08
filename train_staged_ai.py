import numpy as np
import pickle
from staged_learning_system import StagedNeuralNetwork

print("Loading dataset...")
with open("dataset_features.pkl", "rb") as f:
    data_inputs2 = pickle.load(f)

features_STDs = np.std(a=data_inputs2, axis=0)
data_inputs = data_inputs2[:, features_STDs > 50]

with open("outputs.pkl", "rb") as f:
    data_outputs = pickle.load(f)

print(f"Dataset loaded: {data_inputs.shape[0]} samples, {data_inputs.shape[1]} features")
print(f"Classes: {len(np.unique(data_outputs))}")

hidden_layers = [150, 60]
output_size = 4

print("\nInitializing Staged Learning AI...")
print("This AI will learn through 7 developmental stages:")
print("1. Baby Steps")
print("2. Toddler") 
print("3. Pre-K")
print("4. Elementary")
print("5. Teen")
print("6. Scholar")
print("7. Thinker")
print("\nEach stage prioritizes QUALITY and UNDERSTANDING over quantity.")

ai = StagedNeuralNetwork(
    input_size=data_inputs.shape[1],
    max_hidden_sizes=hidden_layers,
    output_size=output_size
)

print("\nBeginning developmental journey...\n")

history = ai.learn_through_stages(
    data_inputs=data_inputs,
    data_outputs=data_outputs,
    population_size=12
)

print("\n" + "="*60)
print("DEVELOPMENTAL JOURNEY COMPLETE")
print("="*60)

for stage_result in history:
    status = "✓ PASSED" if stage_result['passed'] else "✗ NEEDS WORK"
    print(f"{stage_result['stage']:12} - Understanding: {stage_result['metrics']['understanding']:.3f} - {status}")

with open('trained_staged_ai.pkl', 'wb') as f:
    pickle.dump(ai, f)
    
print("\nTrained AI saved to trained_staged_ai.pkl")
print("\nThis AI learned through quality-focused understanding,")
print("not quantity-focused memorization like typical LLMs.")
