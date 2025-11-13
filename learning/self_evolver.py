# core/self_evolver.py
import os, json, ast, random, numpy as np
from datetime import datetime
from core.genetic_trainer import GeneticTrainer
from core.neural_network import NeuralNetwork

class SelfEvolver:
    def __init__(self, trainer: GeneticTrainer, network: NeuralNetwork):
        self.trainer = trainer
        self.network = network
        self.generation = 0
        self.improvement_log = []

    def analyze_code(self):
        """Parse own code and suggest improvements"""
        with open('core/self_evolver.py', 'r') as f:
            tree = ast.parse(f.read())
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        return [f.name for f in functions]

    def mutate_self(self):
        """Apply small, safe mutations to own code"""
        if random.random() > 0.7:  # 30% chance per generation
            with open('core/self_evolver.py', 'a') as f:
                f.write(f"\n# Auto-improved at {datetime.now()}\n")
                f.write("def reflect_on_performance(self):\n")
                f.write("    print(f'Generation {self.generation}: Fitness {self.trainer.best_fitness:.4f}')\n")
            print("AI improved its own code!")

    def evolve_architecture(self):
        """Grow/shrink network based on performance"""
        if self.trainer.best_fitness > 0.8 and len(self.network.layers) < 5:
            self.network.add_layer(nodes=64)
            print("AI grew a new neural layer!")
        elif self.trainer.best_fitness < 0.3:
            self.network.prune(0.1)
            print("AI pruned weak connections.")

    def run_evolution_cycle(self):
        self.generation += 1
        self.mutate_self()
        self.evolve_architecture()
        self.improvement_log.append({
            "gen": self.generation,
            "fitness": self.trainer.best_fitness,
            "timestamp": datetime.now().isoformat()
        })
        with open("knowledge/self_evolution.json", "w") as f:
            json.dump(self.improvement_log, f, indent=2)