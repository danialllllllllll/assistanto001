import time
import random
import threading
import numpy as np
from queue import Queue
from datetime import datetime
import json
import os

from learning.neural_network import ProgressiveNeuralNetwork
from learning.web_learning import AdvancedWebLearning
from learning.node_visualizer import NodeVisualizer
from learning.learning_node_manager import LearningNodeManager
from learning.self_evolver import SelfEvolver
from learning.code_rewriter import CodeRewriter
from learning.realtime_visualizer import RealtimeVisualizer
from learning.genetic_learning import GeneticLearning
from learning.stage_algorithms import get_algorithm_for_stage

class WhimsyTrainer:
    def __init__(self):
        self.stage = 0
        self.iteration = 0
        self.generation = 0
        self.understanding = 0.0
        self.confidence = 0.0
        self.accuracy = 0.0
        self.running = True
        self.update_queue = Queue()

        self.stages = [
            {"name": "Baby Steps", "nodes": 0.10, "target_understanding": 0.25},
            {"name": "Toddler", "nodes": 0.25, "target_understanding": 0.40},
            {"name": "Pre-K", "nodes": 0.40, "target_understanding": 0.55},
            {"name": "Elementary", "nodes": 0.60, "target_understanding": 0.70},
            {"name": "Teen", "nodes": 0.80, "target_understanding": 0.85},
            {"name": "Scholar", "nodes": 0.95, "target_understanding": 0.95},
            {"name": "Thinker", "nodes": 1.00, "target_understanding": 0.99}
        ]

        self.network = ProgressiveNeuralNetwork(
            input_size=102,
            hidden_sizes=[150, 60],
            output_size=4
        )

        self.web_learner = AdvancedWebLearning()
        self.node_manager = LearningNodeManager(initial_nodes=30)
        self.visualizer = NodeVisualizer()
        self.realtime_viz = RealtimeVisualizer()
        self.evolver = SelfEvolver()
        self.code_rewriter = CodeRewriter()
        self.genetic_learner = GeneticLearning(population_size=15)

        self.knowledge = []
        self.evolution_events = []
        self.code_rewrites = []
        self.genetic_patterns = []
        self.current_learning_topic = None
        self.current_learning_understanding = 0.0
        
        self.realtime_viz.initialize_from_network(self.network)

        self.X_train = np.random.randn(200, 102)
        self.y_train = np.random.randint(0, 4, 200)

    def get_current_stage(self):
        return self.stages[min(self.stage, len(self.stages) - 1)]

    def learn_topic(self, topic: str) -> dict:
        """Learn a topic through chat interface until 99% understanding"""
        stage_algo = get_algorithm_for_stage(self.stage)
        self.current_learning_topic = topic
        
        print(f"\n[LEARNING] {self.get_current_stage()['name']} stage learning: {topic}")
        
        # Get knowledge from web sources using search_and_learn
        learned_data = self.web_learner.search_and_learn(topic, depth=stage_algo.min_sources)
        
        # Convert learned_data to knowledge_items format for stage algorithm
        knowledge_items = []
        if learned_data and learned_data.get('processed_knowledge'):
            for source, data in learned_data['processed_knowledge'].items():
                knowledge_items.append({
                    'source': source,
                    'content': data,
                    'confidence': data.get('quality_score', 0.5) if isinstance(data, dict) else 0.5,
                    'source_type': source,
                    'sources': learned_data.get('sources', [])
                })
        
        # Process knowledge through stage-specific algorithm
        if knowledge_items:
            result = stage_algo.process_knowledge(knowledge_items)
            self.understanding = result['understanding']
            self.current_learning_understanding = self.understanding
            
            # Store processed knowledge
            knowledge_entry = {
                "topic": topic,
                "stage": self.get_current_stage()['name'],
                "understanding": self.understanding,
                "method": result.get('method', ''),
                "insights": result.get('insights', []),
                "timestamp": datetime.now().isoformat(),
                "raw_items": knowledge_items,
                "learned_data": learned_data
            }
            self.knowledge.append(knowledge_entry)
            
            # Log learning process
            print(f"[LEARNING] Understanding: {self.understanding*100:.1f}% (target: 99%)")
            print(f"[LEARNING] Method: {result.get('method', 'N/A')}")
            print(f"[LEARNING] Sources: {', '.join(learned_data.get('sources', []))}")
            if result.get('insights'):
                print(f"[LEARNING] Insights: {result['insights'][0]}")
        else:
            self.understanding = 0.0
            print(f"[LEARNING] Failed to acquire knowledge about {topic}")
            
        # Check for stage advancement
        stage_advanced = False
        if self.understanding >= 0.99:
            old_stage = self.get_current_stage()['name']
            self.advance_stage()
            new_stage = self.get_current_stage()['name']
            stage_advanced = True
            print(f"\n🎉 STAGE ADVANCEMENT: {old_stage} → {new_stage}\n")
            
        return {
            "topic": topic,
            "understanding": self.understanding,
            "target": 0.99,
            "stage": self.get_current_stage()['name'],
            "knowledge_items": len(knowledge_items),
            "stage_advanced": stage_advanced,
            "method": result.get('method', '') if knowledge_items else '',
            "learning_complete": self.understanding >= 0.99,
            "sources": learned_data.get('sources', []) if learned_data else []
        }
    
    def advance_stage(self):
        """Advance to next developmental stage"""
        if self.stage < len(self.stages) - 1:
            self.stage += 1
            # Reset understanding for new stage
            self.understanding = 0.0
            self.current_learning_understanding = 0.0
            return True
        return False

    def train_iteration(self):
        """Background training - only updates network architecture, NOT understanding/learning"""
        stage_info = self.get_current_stage()
        self.network.set_stage_activation(stage_info['nodes'])

        batch_size = 64
        indices = np.random.choice(len(self.X_train), batch_size, replace=False)
        X_batch = self.X_train[indices]
        y_batch = self.y_train[indices]

        self.network.forward(X_batch, training=True)
        self.realtime_viz.update_from_forward_pass(self.network, self.network.activations)
        self.network.backward(X_batch, y_batch, learning_rate=0.001)

        predictions = self.network.predict(self.X_train)
        confidences = self.network.get_confidence(self.X_train)

        correct = predictions == self.y_train
        self.accuracy = float(np.mean(correct))
        self.confidence = float(np.mean(confidences[correct]) if np.any(correct) else 0)

        self.iteration += 1

        if self.iteration % 10 == 0:
            self.generation += 1
            self.update_visualization()

        if self.iteration % 50 == 0:
            self.rewrite_own_code()
            
        # NOTE: Stage advancement now ONLY happens through learn_topic() reaching 99% understanding

    def autonomous_evolve(self):
        prev_fitness = self.evolver.fitness_history[-1] if self.evolver.fitness_history else 0
        current_fitness = self.understanding

        self.evolver.fitness_history.append(current_fitness)
        self.evolver.generation_counter = self.generation

        if self.evolver.should_evolve(self.generation):
            suggestions = self.evolver.generate_evolution_suggestions(
                current_fitness, prev_fitness, self.generation
            )

            for sugg in suggestions:
                event = {
                    "generation": self.generation,
                    "type": sugg.change_type,
                    "change": f"{sugg.target}: {sugg.old_value} -> {sugg.new_value}",
                    "reason": sugg.reasoning,
                    "timestamp": datetime.now().isoformat()
                }
                self.evolution_events.append(event)
                self.update_queue.put({"type": "evolution", "event": event})
                print(f"[EVOLUTION] {sugg.target}: {sugg.old_value} -> {sugg.new_value}")

    def web_learn(self):
        """Disabled - learning now happens only through chat interface"""
        pass

    def genetic_evolution(self):
        """Evolve learning strategies using genetic algorithms"""
        try:
            evolution_result = self.genetic_learner.evolve_generation(
                self.network,
                self.X_train,
                self.y_train
            )
            
            genetic_event = {
                "iteration": self.iteration,
                "generation": evolution_result['generation'],
                "best_fitness": evolution_result['best_fitness'],
                "avg_fitness": evolution_result['avg_fitness'],
                "diversity": evolution_result['population_diversity'],
                "timestamp": datetime.now().isoformat()
            }
            self.genetic_patterns.append(genetic_event)
            self.update_queue.put({"type": "genetic", "event": genetic_event})
            
            print(f"[GENETIC] Gen {evolution_result['generation']}: "
                  f"Best={evolution_result['best_fitness']:.4f}, "
                  f"Avg={evolution_result['avg_fitness']:.4f}, "
                  f"Diversity={evolution_result['population_diversity']:.4f}")
            
            patterns = self.genetic_learner.get_adaptive_patterns_summary()
            if patterns['recent_patterns']:
                for pattern in patterns['recent_patterns'][-2:]:
                    self.realtime_viz.track_genetic_pattern(
                        pattern['type'],
                        pattern['description'],
                        pattern['effectiveness']
                    )
                    print(f"[PATTERN] {pattern['type']}: {pattern['description']}")
        except Exception as e:
            print(f"Genetic evolution error: {e}")
            import traceback
            traceback.print_exc()

    def rewrite_own_code(self):
        """Rewrite own Python code every 50 iterations for autonomous optimization"""
        if self.stage < 2:
            return
        
        try:
            metrics = {
                'accuracy': self.accuracy,
                'understanding': self.understanding,
                'confidence': self.confidence,
                'iteration': self.iteration
            }
            
            analysis = self.code_rewriter.analyze_performance(metrics)
            
            if analysis['needs_optimization']:
                print(f"\n[CODE EVOLUTION] Performance trend: {analysis['performance_trend']}")
                modifications = self.code_rewriter.generate_code_modifications(analysis, self.iteration)
                
                if modifications:
                    results = self.code_rewriter.apply_modifications(modifications)
                    
                    for mod_dict in results['applied']:
                        rewrite_event = {
                            "iteration": self.iteration,
                            "type": "code_rewrite",
                            "file": mod_dict['file_path'],
                            "description": mod_dict['description'],
                            "timestamp": datetime.now().isoformat()
                        }
                        self.code_rewrites.append(rewrite_event)
                        self.update_queue.put({"type": "code_rewrite", "event": rewrite_event})
                    
                    print(f"[CODE REWRITE] Applied {len(results['applied'])} modifications, {len(results['failed'])} failed")
        except Exception as e:
            print(f"Code rewrite error: {e}")
            import traceback
            traceback.print_exc()

    def update_visualization(self):
        nodes_data = []
        for node_id, node in self.node_manager.nodes.items():
            if not node.pruned:
                nodes_data.append({
                    'id': node_id,
                    'layer': node.layer,
                    'specialization': node.specialization,
                    'fitness': node.fitness,
                    'age': node.age
                })

        connections_data = []
        for node_id in list(self.node_manager.nodes.keys())[:20]:
            connected = self.node_manager.get_node_connections(node_id)
            for conn_id in connected[:3]:
                connections_data.append({
                    'source': node_id,
                    'target': conn_id,
                    'weight': random.uniform(0.3, 1.0)
                })

        topology = self.visualizer.create_network_topology_data(nodes_data, connections_data)

        try:
            os.makedirs('data', exist_ok=True)
            with open('data/network_topology.json', 'w') as f:
                json.dump(topology, f)
            
            self.realtime_viz.save_visualization_state('data/realtime_viz.json')
        except Exception as e:
            print(f"Viz save error: {e}")

    def get_state(self):
        stage_info = self.get_current_stage()
        return {
            "stage": stage_info['name'],
            "stage_index": self.stage,
            "iteration": self.iteration,
            "generation": self.generation,
            "understanding": round(self.understanding, 4),
            "confidence": round(self.confidence, 4),
            "accuracy": round(self.accuracy, 4),
            "nodes_active": stage_info['nodes'],
            "knowledge_count": len(self.knowledge),
            "evolution_count": len(self.evolution_events),
            "target_understanding": stage_info['target_understanding']
        }

    def train_loop(self):
        print("\nWHIMSY TRAINING STARTED\n")

        while self.running and self.stage < len(self.stages):
            try:
                self.train_iteration()
                time.sleep(0.02)
            except Exception as e:
                print(f"Training error: {e}")
                import traceback
                traceback.print_exc()

        print("\nWHIMSY TRAINING COMPLETE - THINKER STAGE REACHED\n")

trainer = WhimsyTrainer()
