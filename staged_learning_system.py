import numpy as np
import pickle
import json
from datetime import datetime

class DevelopmentalStage:
    """Represents a developmental stage with specific learning characteristics"""
    def __init__(self, name, description, active_nodes_percent, understanding_threshold, 
                 learning_rate, quality_weight, max_iterations):
        self.name = name
        self.description = description
        self.active_nodes_percent = active_nodes_percent
        self.understanding_threshold = understanding_threshold
        self.learning_rate = learning_rate
        self.quality_weight = quality_weight
        self.max_iterations = max_iterations
        
class StagedNeuralNetwork:
    """Progressive neural network that learns through developmental stages"""
    
    def __init__(self, input_size, max_hidden_sizes, output_size):
        self.input_size = input_size
        self.max_hidden_sizes = max_hidden_sizes
        self.output_size = output_size
        self.current_stage = 0
        
        self.stages = [
            DevelopmentalStage(
                name="Baby Steps",
                description="Incoherent thought, minimalistic communication. Learning basic patterns.",
                active_nodes_percent=0.10,
                understanding_threshold=0.30,
                learning_rate=0.01,
                quality_weight=0.3,
                max_iterations=50
            ),
            DevelopmentalStage(
                name="Toddler",
                description="Partial coherent thought, improved memory, basic understanding.",
                active_nodes_percent=0.25,
                understanding_threshold=0.45,
                learning_rate=0.02,
                quality_weight=0.4,
                max_iterations=75
            ),
            DevelopmentalStage(
                name="Pre-K",
                description="Conscious of surroundings, begins to think and ponder.",
                active_nodes_percent=0.40,
                understanding_threshold=0.60,
                learning_rate=0.03,
                quality_weight=0.5,
                max_iterations=100
            ),
            DevelopmentalStage(
                name="Elementary",
                description="Questions surroundings, prioritizes understanding over quantity.",
                active_nodes_percent=0.60,
                understanding_threshold=0.70,
                learning_rate=0.04,
                quality_weight=0.7,
                max_iterations=150
            ),
            DevelopmentalStage(
                name="Teen",
                description="Prioritizes learning to entirety, quality over quantity, develops personality.",
                active_nodes_percent=0.80,
                understanding_threshold=0.80,
                learning_rate=0.05,
                quality_weight=0.85,
                max_iterations=200
            ),
            DevelopmentalStage(
                name="Scholar",
                description="Masters growth, hyper aware, adapts to biases, 99% truth accuracy.",
                active_nodes_percent=0.95,
                understanding_threshold=0.90,
                learning_rate=0.03,
                quality_weight=0.95,
                max_iterations=250
            ),
            DevelopmentalStage(
                name="Thinker",
                description="Philosophy prioritized, finalized personality, ensures kindness over ego.",
                active_nodes_percent=1.0,
                understanding_threshold=0.95,
                learning_rate=0.02,
                quality_weight=1.0,
                max_iterations=300
            )
        ]
        
        self.initialize_network()
        self.learning_history = []
        
    def initialize_network(self):
        """Initialize weights with full capacity but most nodes inactive"""
        self.weights = []
        prev_size = self.input_size
        
        for hidden_size in self.max_hidden_sizes:
            w = np.random.uniform(-0.1, 0.1, size=(prev_size, hidden_size))
            self.weights.append(w)
            prev_size = hidden_size
            
        self.weights.append(np.random.uniform(-0.1, 0.1, size=(prev_size, self.output_size)))
        
        self.node_activation_scales = []
        for w in self.weights[:-1]:
            self.node_activation_scales.append(np.ones(w.shape[1]))
        
    def activate_nodes_for_stage(self, stage_idx):
        """Activate percentage of nodes based on developmental stage"""
        stage = self.stages[stage_idx]
        
        for i, scales in enumerate(self.node_activation_scales):
            num_nodes = len(scales)
            num_active = int(num_nodes * stage.active_nodes_percent)
            
            scales[:] = 0.0
            active_indices = np.random.choice(num_nodes, num_active, replace=False)
            scales[active_indices] = 1.0
            
        print(f"\n{'='*60}")
        print(f"STAGE: {stage.name}")
        print(f"{'='*60}")
        print(f"{stage.description}")
        print(f"Active nodes: {stage.active_nodes_percent*100:.0f}%")
        print(f"Understanding threshold: {stage.understanding_threshold*100:.0f}%")
        print(f"{'='*60}\n")
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def forward_pass(self, x):
        """Forward pass with scaled nodes (inactive nodes output 0)"""
        activation = x
        
        for i, (w, scales) in enumerate(zip(self.weights[:-1], self.node_activation_scales)):
            activation = np.dot(activation, w)
            activation = self.sigmoid(activation) * scales
        
        activation = np.dot(activation, self.weights[-1])
        activation = self.sigmoid(activation)
        return activation
    
    def test_understanding(self, data_inputs, data_outputs, stage):
        """Test true understanding, not just memorization"""
        correct_confident = 0
        correct_total = 0
        confidence_sum = 0
        
        for i in range(len(data_inputs)):
            output = self.forward_pass(data_inputs[i])
            predicted = np.argmax(output)
            actual = int(data_outputs[i])
            confidence = output[predicted]
            
            if predicted == actual:
                correct_total += 1
                confidence_sum += confidence
                
                if confidence > 0.7:
                    correct_confident += 1
        
        accuracy = correct_total / len(data_inputs) if len(data_inputs) > 0 else 0
        avg_confidence = confidence_sum / correct_total if correct_total > 0 else 0
        
        understanding_score = (accuracy * stage.quality_weight + 
                             avg_confidence * (1 - stage.quality_weight))
        
        return {
            'accuracy': accuracy,
            'confidence': avg_confidence,
            'understanding': understanding_score,
            'confident_correct': correct_confident / len(data_inputs) if len(data_inputs) > 0 else 0
        }
    
    def quiz_until_understood(self, data_inputs, data_outputs, stage, population):
        """Quiz self until topic is understood - quality over quantity"""
        print(f"\nQuizzing for understanding (not just memorization)...")
        
        best_understanding = 0
        iterations_without_improvement = 0
        max_no_improvement = 20
        
        for iteration in range(stage.max_iterations):
            fitness_scores = []
            
            for individual in population:
                self.load_weights(individual)
                metrics = self.test_understanding(data_inputs, data_outputs, stage)
                
                understanding_penalty = 0
                if metrics['confidence'] < 0.6:
                    understanding_penalty = 0.2
                
                fitness = metrics['understanding'] - understanding_penalty
                fitness_scores.append(fitness)
            
            best_idx = np.argmax(fitness_scores)
            current_best = fitness_scores[best_idx]
            
            if current_best > best_understanding:
                best_understanding = current_best
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
            
            if iteration % 10 == 0:
                self.load_weights(population[best_idx])
                metrics = self.test_understanding(data_inputs, data_outputs, stage)
                print(f"Iteration {iteration}: Understanding={metrics['understanding']:.3f}, "
                      f"Accuracy={metrics['accuracy']:.3f}, Confidence={metrics['confidence']:.3f}")
            
            if best_understanding >= stage.understanding_threshold and iterations_without_improvement > 5:
                print(f"\n✓ Understanding achieved after {iteration} iterations!")
                break
                
            if iterations_without_improvement >= max_no_improvement:
                print(f"\nPlateau reached at {best_understanding:.3f} understanding")
                break
            
            population = self.evolve_population(population, fitness_scores, stage.learning_rate)
        
        self.load_weights(population[best_idx])
        final_metrics = self.test_understanding(data_inputs, data_outputs, stage)
        
        return final_metrics, population[best_idx]
    
    def load_weights(self, weight_list):
        """Load weights from a list"""
        for i in range(len(self.weights)):
            self.weights[i] = weight_list[i].copy()
    
    def evolve_population(self, population, fitness_scores, learning_rate):
        """Evolve population with quality-focused selection"""
        fitness_array = np.array(fitness_scores)
        
        num_parents = max(2, len(population) // 3)
        parent_indices = np.argsort(fitness_array)[-num_parents:]
        parents = [population[i] for i in parent_indices]
        
        new_population = parents.copy()
        
        while len(new_population) < len(population):
            p1, p2 = np.random.choice(len(parents), 2, replace=False)
            child = []
            
            for w1, w2 in zip(parents[p1], parents[p2]):
                if np.random.random() < 0.5:
                    child_w = w1.copy()
                else:
                    child_w = w2.copy()
                
                mutation_mask = np.random.random(child_w.shape) < 0.1
                child_w[mutation_mask] += np.random.normal(0, learning_rate, child_w.shape)[mutation_mask]
                child_w = np.clip(child_w, -1, 1)
                
                child.append(child_w)
            
            new_population.append(child)
        
        return new_population[:len(population)]
    
    def learn_through_stages(self, data_inputs, data_outputs, population_size=10):
        """Progress through all developmental stages"""
        print("\n" + "="*60)
        print("DEVELOPMENTAL LEARNING SYSTEM")
        print("Quality Over Quantity - Understanding Over Memorization")
        print("="*60)
        
        population = []
        for _ in range(population_size):
            individual = [w.copy() for w in self.weights]
            population.append(individual)
        
        for stage_idx, stage in enumerate(self.stages):
            self.current_stage = stage_idx
            self.activate_nodes_for_stage(stage_idx)
            
            metrics, best_weights = self.quiz_until_understood(
                data_inputs, data_outputs, stage, population
            )
            
            stage_result = {
                'stage': stage.name,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'passed': metrics['understanding'] >= stage.understanding_threshold
            }
            
            self.learning_history.append(stage_result)
            
            print(f"\n{'='*60}")
            print(f"STAGE COMPLETE: {stage.name}")
            print(f"Understanding Score: {metrics['understanding']:.3f} (threshold: {stage.understanding_threshold})")
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"Confidence: {metrics['confidence']:.3f}")
            print(f"Status: {'PASSED ✓' if stage_result['passed'] else 'NEEDS MORE WORK ✗'}")
            print(f"{'='*60}\n")
            
            if not stage_result['passed']:
                print(f"⚠ Stage {stage.name} not fully understood. Continuing but flagged for review.")
            
            population = [[w.copy() for w in self.weights] for _ in range(population_size)]
        
        self.save_learning_history()
        return self.learning_history
    
    def save_learning_history(self):
        """Save the learning journey"""
        with open('learning_history.json', 'w') as f:
            json.dump(self.learning_history, f, indent=2)
        print("\nLearning history saved to learning_history.json")
