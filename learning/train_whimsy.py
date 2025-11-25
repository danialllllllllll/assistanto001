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
        self.evolver = SelfEvolver()

        self.knowledge = []
        self.evolution_events = []

        self.X_train = np.random.randn(200, 102)
        self.y_train = np.random.randint(0, 4, 200)

    def get_current_stage(self):
        return self.stages[min(self.stage, len(self.stages) - 1)]

    def train_iteration(self):
        stage_info = self.get_current_stage()
        self.network.set_stage_activation(stage_info['nodes'])

        batch_size = 32
        indices = np.random.choice(len(self.X_train), batch_size, replace=False)
        X_batch = self.X_train[indices]
        y_batch = self.y_train[indices]

        self.network.forward(X_batch, training=True)
        self.network.backward(X_batch, y_batch, learning_rate=0.001)

        predictions = self.network.predict(self.X_train)
        confidences = self.network.get_confidence(self.X_train)

        correct = predictions == self.y_train
        self.accuracy = float(np.mean(correct))
        self.confidence = float(np.mean(confidences[correct]) if np.any(correct) else 0)
        self.understanding = (self.accuracy * 0.6 + self.confidence * 0.4)

        self.iteration += 1

        if self.iteration % 5 == 0:
            self.generation += 1
            self.autonomous_evolve()

        if self.iteration % 20 == 0:
            self.web_learn()

        if self.iteration % 10 == 0:
            self.update_visualization()

        if self.understanding >= stage_info['target_understanding'] and self.stage < len(self.stages) - 1:
            self.stage += 1
            msg = f"ADVANCED TO {self.stages[self.stage]['name']}"
            self.update_queue.put({"type": "stage_change", "message": msg})
            print(f"\n{msg}\n")

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
        if self.stage < 3:
            return

        topics = ['neural networks', 'machine learning', 'artificial intelligence',
                  'deep learning', 'cognitive science', 'philosophy of mind']
        topic = random.choice(topics)

        try:
            learned = self.web_learner.search_and_learn(topic)
            if learned and learned.get('confidence', 0) > 0.5:
                knowledge_item = {
                    "topic": topic,
                    "stage": self.get_current_stage()['name'],
                    "confidence": learned['confidence'],
                    "sources": len(learned['sources']),
                    "timestamp": datetime.now().isoformat()
                }
                self.knowledge.append(knowledge_item)
                self.update_queue.put({"type": "knowledge", "item": knowledge_item})
                print(f"[WEB LEARN] {topic}: {learned['confidence']:.2f} confidence from {len(learned['sources'])} sources")
        except Exception as e:
            print(f"Web learning error: {e}")

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
