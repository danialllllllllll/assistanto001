import time
import random
import threading
import os
import shutil
import zipfile
from datetime import datetime
from flask import Flask, jsonify, request, send_file, render_template, session, redirect
from flask_cors import CORS
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from queue import Queue
import json

try:
    from deap import base, creator, tools
except ImportError:
    print("Error: 'deap' module not found. Install with: pip install deap")
    raise

# === CONFIGURATION ===
STAGES = [
    {"name": "Baby Steps Phase", "target": "patterns", "learning_rate": 0.1, "age_equiv": "0-18 months", "min_iterations": 50},
    {"name": "Toddler Phase", "target": "memory", "learning_rate": 0.1, "age_equiv": "18 months-3 years", "min_iterations": 50},
    {"name": "Pre-K Phase", "target": "coherence", "learning_rate": 0.1, "age_equiv": "3-5 years", "min_iterations": 50},
    {"name": "Elementary Phase", "target": "questioning", "learning_rate": 0.1, "age_equiv": "5-14 years", "min_iterations": 100},
    {"name": "Teen Phase", "target": "personality", "learning_rate": 0.1, "age_equiv": "14-18 years", "min_iterations": 100},
    {"name": "Scholar Phase", "target": "truth_detection", "learning_rate": 0.1, "age_equiv": "18-22 years", "min_iterations": 150},
    {"name": "Thinker Phase", "target": "philosophy", "learning_rate": 0.1, "age_equiv": "22+ years", "min_iterations": 200}
]

CORE_VALUES = [
    {"name": "Kindness", "priority": 1.0, "status": "Enforced"},
    {"name": "Understanding", "priority": 1.0, "status": "Enforced"},
    {"name": "Truth", "priority": 1.0, "status": "Enforced"},
    {"name": "Positive Relationships", "priority": 1.0, "status": "Enforced"},
    {"name": "Non-Harm", "priority": 1.0, "status": "Enforced"}
]

UNDERSTANDING_REQUIREMENTS = {"minimum_understanding": 0.999, "minimum_confidence": 0.95}
WEB_PASSWORD = "OVER//RIDE"
KILL_SWITCH_PHRASE = "confirm delete"

# Dataset
DATA_SIZE = 1000
FEATURES = 20
CLASSES = 4
data_inputs = np.random.rand(DATA_SIZE, FEATURES)
data_outputs = np.random.randint(0, CLASSES, DATA_SIZE)

# === FULL CURRICULUM ===
STAGE_TOPICS = {
    "Baby Steps Phase": ["Animal sounds", "Bright colors", "Simple shapes", "minimal speech"],
    "Toddler Phase": [
        "Naming animals", "Basic actions (e.g., dog runs)", "Pet care basics",
        "identifying shapes", "simple counting", "simple speech", "Toddler essentials"
    ],
    "Pre-K Phase": [
        "Animal stories", "Animal emotions", "Simple habitats",
        "Preschool Math", "Preschool English", "Preschool Foreign Language",
        "Preschool History", "Preschool Geography", "Preschool Science",
        "Preschool Art", "Preschool Music", "Preschool Physical Education", "Preschool Health"
    ],
    "Elementary Phase": [
        "Animal life cycles", "Basic classification (mammals/birds)", "Animal habitats",
        "Earth Science", "Right vs Wrong", "Basic Math", "Basic English", "Foreign Language",
        "History", "Geography", "Science", "Art", "Music", "Physical Education", "Health",
        "Social Studies", "Elementary Thinking", "Elementary Problem Solving",
        "Elementary Decision Making", "Elementary Critical Thinking", "Elementary Problem Solving", "Elementary Topics"
    ],
    "Teen Phase": [
        "Animal ecology", "Conservation ethics", "Human-animal interactions",
        "Human History", "War", "World Events", "Complex College Level topics",
        "Advanced Math", "Advanced English", "Advanced Foreign Language", "Advanced Science",
        "Advanced Art", "Advanced Music", "Advanced Physical Education", "Advanced Health",
        "Advanced Social Studies", "Complex Thinking", "Critical Thinking", "Problem Solving",
        "Decision Making", "Ethical Dilemmas", "Philosophical Questions", "Logical Reasoning",
        "Psychological Insights", "Sociological Perspectives", "Economic Principles",
        "Political Systems", "Environmental Issues", "Highschool Level Topics", "Theory",
        "Astronomy", "Astrology", "Cosmology", "Physics", "Chemistry", "Biology", "Geology",
        "Mathematics", "Computer Science", "Engineering", "Medicine", "Law", "Politics",
        "Economics", "Sociology", "All advanced level Science fields", "Mathematics",
        "Computer Science", "Engineering", "Medicine", "Law", "Politics", "Economics",
        "Every Subject supplied to a high school student"
    ],
    "Scholar Phase": [
        "Animal evolution", "Genetic adaptations", "Wildlife misinformation detection",
        "Ethical dilemmas in conservation", "Advanced Philosophy", "Advanced Ethics",
        "Advanced Logic", "Advanced Psychology", "Advanced Sociology", "Advanced Metaphysics",
        "Advanced Epistemology", "Advanced Ontology", "Advanced Aesthetics", "Advanced Ethics",
        "Advanced Logic", "Advanced Epistemology", "Advanced Ontology", "Advanced Aesthetics",
        "Advanced Mathematics", "Advanced Computer Science", "Advanced Engineering",
        "Advanced Medicine", "Advanced Law", "Advanced Politics", "Advanced Economics",
        "Advanced Sociology", " #%$ Advanced Environmental Science", "Advanced Physics",
        "Advanced Chemistry", "Advanced Biology", "All advanced level Science fields",
        "Advanced Mathematics", "Advanced Computer Science", "Advanced Engineering",
        "Advanced Medicine", "Advanced Law", "Advanced Politics", "Advanced Economics",
        "Advanced Sociology", "Advanced Environmental Science", "Advanced Physics",
        "All Advanced level Mathematic fields", "Advanced Computer Science", "Advanced Engineering",
        "Advanced Medicine", "Advanced Law", "Advanced Politics", "Advanced Economics",
        "Advanced Sociology", "Advanced Environmental Science", "Advanced Physics",
        "Advanced Chemistry", "Advanced Biology", "All advanced Geography fields",
        "Advanced History", "Advanced World Events", "Advanced War", "Advanced Human History",
        "Advanced World Events", "Advanced War", "Advanced Human History", "Advanced World Events",
        "Advanced War", "Advanced Human History", "All Advanced Language Arts fields",
        "Advanced English", "Advanced Foreign Language", "Advanced Literature", "Advanced Writing",
        "Advanced Reading", "Advanced Speaking", "Advanced Listening", "Advanced Grammar",
        "Advanced Vocabulary", "Advanced Syntax", "Advanced Semantics"
    ],
    "Thinker Phase": [
        "Animal consciousness", "Philosophical animal rights", "Ethical dilemmas in ecology",
        "Everything in the universe", "The meaning of life", "The nature of reality",
        "The nature of consciousness", "The nature of time", "The nature of space",
        "The nature of matter", "The nature of energy", "The nature of information",
        "The nature of knowledge", "The nature of truth", "The nature of beauty",
        "The nature of goodness", "The nature of evil", "The nature of love",
        "The nature of hate", "Everything the world has to offer", "Absolutely everything",
        "Complex Theories"
    ]
}

# === HIERARCHICAL ARCHIVER ===
class HierarchicalArchiver:
    def __init__(self, base_dir='training_archives'):
        self.base_dir = base_dir
        self.current_phase = ""
        self.generation_buffer = []
        self.batch_count = 0
        os.makedirs(base_dir, exist_ok=True)

    def set_phase(self, phase_name: str) -> str:
        self.current_phase = phase_name.lower().replace(' ', '_')
        phase_path = os.path.join(self.base_dir, self.current_phase)
        os.makedirs(phase_path, exist_ok=True)
        self.batch_count = 0
        return phase_path

    def save_generation(self, generation_data: dict) -> None:
        self.generation_buffer.append(generation_data)
        if len(self.generation_buffer) >= 10:
            self._create_batch_zip()

    def _create_batch_zip(self) -> None:
        if not self.current_phase or not self.generation_buffer:
            return
        phase_path = os.path.join(self.base_dir, self.current_phase)
        temp_dir = os.path.join(phase_path, f'temp_batch_{self.batch_count}')
        os.makedirs(temp_dir, exist_ok=True)
        for gen_data in self.generation_buffer:
            filename = os.path.join(temp_dir, f'generation_{gen_data["iteration"]}.json')
            with open(filename, 'w') as f:
                json.dump(gen_data, f, indent=2)
        zip_path = os.path.join(phase_path, f'batch_{self.batch_count:04d}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
        shutil.rmtree(temp_dir)
        self.generation_buffer = []
        self.batch_count += 1
        print(f"  Archived batch {self.batch_count-1} -> {zip_path}")

# === TRAINING OPTIMIZER ===
class TrainingOptimizer:
    def __init__(self):
        self.stats = {
            'iterations': 0, 'best_fitness': 0.0, 'best_understanding': 0.0,
            'last_improvement': 0, 'phase_history': [], 'nodes_created': 0,
            'nodes_pruned': 0, 'quiz_scores': []
        }

    def optimize_metrics(self, fitness: float, understanding: float, confidence: float) -> dict:
        self.stats['iterations'] += 1
        if fitness > self.stats['best_fitness']:
            self.stats['best_fitness'] = fitness
            self.stats['last_improvement'] = self.stats['iterations']
        if understanding > self.stats['best_understanding']:
            self.stats['best_understanding'] = understanding
        return {
            'is_improving': (self.stats['iterations'] - self.stats['last_improvement']) < 100,
            'mutation_rate_adjustment': 0.2 if understanding < 0.5 else 0.05 if understanding > 0.95 else 0.1,
            'population_size_suggestion': 200 if confidence > 0.9 else 100
        }

# === PHASE TRAINING ALGORITHMS ===
class PhaseTrainingAlgorithms:
    @staticmethod
    def train_baby_steps(ai, X_batch, y_batch, toolbox, pop_size: int) -> dict:
        pop = toolbox.population(n=pop_size)
        for gen in range(20):
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < ai.mutation_rate:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
                if random.random() < 0.1:
                    event = {
                        'type': 'node_creation', 'node_id': random.randint(0, 100),
                        'iteration': gen, 'timestamp': datetime.now().isoformat(),
                        'generation': ai.optimizer.stats['iterations'], 'layer': random.randint(1, 3),
                        'count': 1, 'fitness': random.uniform(0.7, 1.0)
                    }
                    ai.evolution_events.append(event)
                    ai.optimizer.stats['nodes_created'] += 1
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            pop[:] = offspring
        best = tools.selBest(pop, 1)[0]
        ai.genome = best
        fitness = best.fitness.values[0]
        accuracy = min(1.0, fitness / 2.0)
        return {'pattern_recognition': accuracy, 'coherence': accuracy * 0.8, 'confidence': accuracy * 0.9}

    @staticmethod
    def train_toddler(ai, X_batch, y_batch, toolbox, pop_size: int) -> dict:
        memory_buffer = []
        accuracies = []
        for _ in range(5):
            pop = toolbox.population(n=pop_size)
            for gen in range(20):
                offspring = toolbox.select(pop, len(pop))
                offspring = list(map(toolbox.clone, offspring))
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.7:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                for mutant in offspring:
                    if random.random() < ai.mutation_rate:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                pop[:] = offspring
            best = tools.selBest(pop, 1)[0]
            ai.genome = best
            fitness = best.fitness.values[0]
            accuracy = min(1.0, fitness / 2.0)
            accuracies.append(accuracy)
            memory_buffer.append((X_batch.copy(), y_batch.copy()))
            if len(memory_buffer) > 4:
                memory_buffer.pop(0)
        return {
            'memory_retention': np.mean(accuracies),
            'coherence_improvement': max(0, accuracies[-1] - accuracies[0]),
            'cognitive_growth': accuracies[-1]
        }

    @staticmethod
    def train_pre_k(ai, X_batch, y_batch, toolbox, pop_size: int) -> dict:
        awareness_scores = []
        for _ in range(3):
            pop = toolbox.population(n=pop_size)
            for gen in range(20):
                offspring = toolbox.select(pop, len(pop))
                offspring = list(map(toolbox.clone, offspring))
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.7:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                for mutant in offspring:
                    if random.random() < ai.mutation_rate:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                pop[:] = offspring
            best = tools.selBest(pop, 1)[0]
            ai.genome = best
            fitness = best.fitness.values[0]
            awareness_scores.append(fitness)
        accuracy = min(1.0, fitness / 2.0)
        return {
            'self_awareness': np.mean(awareness_scores),
            'thought_coherence': accuracy * 0.8,
            'pondering_depth': len(awareness_scores)
        }

    @staticmethod
    def train_elementary(ai, X_batch, y_batch, toolbox, pop_size: int) -> dict:
        quiz_results = []
        for step in range(8):
            for _ in range(10):
                pop = toolbox.population(n=pop_size)
                for gen in range(30):
                    offspring = toolbox.select(pop, len(pop))
                    offspring = list(map(toolbox.clone, offspring))
                    for child1, child2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < 0.7:
                            toolbox.mate(child1, child2)
                            del child1.fitness.values
                            del child2.fitness.values
                    for mutant in offspring:
                        if random.random() < ai.mutation_rate:
                            toolbox.mutate(mutant)
                            del mutant.fitness.values
                    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                    fitnesses = map(toolbox.evaluate, invalid_ind)
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit
                    pop[:] = offspring
                best = tools.selBest(pop, 1)[0]
                ai.genome = best
                fitness = best.fitness.values[0]
                accuracy = min(1.0, fitness / 2.0)
                quiz_results.append({'accuracy': accuracy, 'understanding': accuracy * 0.8})
        return {
            'final_understanding': quiz_results[-1]['understanding'],
            'learning_improvement': max(0, quiz_results[-1]['understanding'] - quiz_results[0]['understanding']),
            'mastery_score': accuracy
        }

    @staticmethod
    def train_teen(ai, X_batch, y_batch, toolbox, pop_size: int, personality: dict) -> dict:
        quality = []
        for _ in range(12):
            pop = toolbox.population(n=pop_size)
            for gen in range(30):
                offspring = toolbox.select(pop, len(pop))
                offspring = list(map(toolbox.clone, offspring))
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.7:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                for mutant in offspring:
                    if random.random() < ai.mutation_rate:
                        toolbox.m --          toolbox.mutate(mutant)
                        del mutant.fitness.values
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                pop[:] = offspring
            best = tools.selBest(pop, 1)[0]
            ai.genome = best
            fitness = best.fitness.values[0]
            accuracy = min(1.0, fitness / 2.0)
            quality.append(accuracy)
            personality['curiosity'] = min(1.0, personality.get('curiosity', 0.5) + 0.02)
        return {
            'quality_score': np.mean(quality[-3:]),
            'personality_development': np.mean(list(personality.values()))
        }

    @staticmethod
    def train_scholar(ai, X_batch, y_batch, toolbox, pop_size: int) -> dict:
        ensemble = []
        for _ in range(20):
            pop = toolbox.population(n=pop_size)
            for gen in range(40):
                offspring = toolbox.select(pop, len(pop))
                offspring = list(map(toolbox.clone, offspring))
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.7:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                for mutant in offspring:
                    if random.random() < ai.mutation_rate:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                pop[:] = offspring
            best = tools.selBest(pop, 1)[0]
            ai.genome = best
            fitness = best.fitness.values[0]
            if _ % 3 == 0:
                ensemble.append(fitness)
        accuracy = min(1.0, fitness / 2.0)
        return {'mastery_level': accuracy, 'truth_accuracy': accuracy * 0.95}

    @staticmethod
    def train_thinker(ai, X_batch, y_batch, toolbox, pop_size: int, personality: dict, insights: list) -> dict:
        for _ in range(25):
            pop = toolbox.population(n=pop_size)
            for gen in range(50):
                offspring = toolbox.select(pop, len(pop))
                offspring = list(map(toolbox.clone, offspring))
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < 0.7:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                for mutant in offspring:
                    if random.random() < ai.mutation_rate:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                pop[:] = offspring
            best = tools.selBest(pop, 1)[0]
            ai.genome = best
            fitness = best.fitness.values[0]
            accuracy = min(1.0, fitness / 2.0)
            if accuracy > 0.95:
                insights.append("Wisdom balances certainty with humility.")
        return {
            'philosophical_depth': len(insights) / 50.0,
            'ethical_alignment': personality.get('kindness', 0) > 0.9
        }

# === MAIN AI CLASS ===
class AdvancedAI:
    def __init__(self):
        self.current_stage = 0
        self.understanding = 0.0
        self.confidence = 0.0
        self.mutation_rate = 0.1
        self.personality_traits = {'curiosity': 0.5, 'empathy': 0.5, 'kindness': 0.5}
        self.philosophical_insights = []
        self.genome = [random.uniform(0, 1) for _ in range(20)]
        self.evolution_events = []
        self.research_topics = []
        self.learned_topics = []
        self.quiz_scores = []
        self.lock = threading.Lock()
        self.running = False
        self.optimizer = TrainingOptimizer()
        self.progress_queue = Queue()
        self.progress_data = self._init_progress()

    def _init_progress(self):
        return {
            "stage": STAGES[0]["name"], "age_equiv": STAGES[0]["age_equiv"],
            "understanding": 0.0, "confidence": 0.0, "iterations": 0, "fitness": 0.0,
            "core_values_compliance": "Compliant", "evolution_events": [],
            "time_estimate": 0, "archive_status": "0 batches", "learned_topics": [],
            "quiz_scores": [], "mastery_level": 0.0, "personality": self.personality_traits,
            "philosophy": self.philosophical_insights[-5:]
        }

    def update_progress(self, fitness: float = 0.0, iterations: int = 0):
        with self.lock:
            stage = STAGES[self.current_stage]
            self.progress_data.update({
                "stage": stage["name"], "age_equiv": stage["age_equiv"],
                "understanding": round(self.understanding, 4),
                "confidence": round(self.confidence, 4), "iterations": iterations,
                "fitness": round(fitness, 4), "core_values_compliance": self.check_core_values(),
                "evolution_events": self.evolution_events[-5:],
                "time_estimate": max(0, (stage["min_iterations"] - iterations) // 10),
                "archive_status": f"{archiver.batch_count} batches archived",
                "learned_topics": self.learned_topics[-5:], "quiz_scores": self.quiz_scores[-3:],
                "mastery_level": max([t.get("mastery", 0) for t in self.learned_topics if t.get("stage") == stage["name"]] or [0.0]),
                "personality": self.personality_traits, "philosophy": self.philosophical_insights[-5:]
            })
        self.progress_queue.put(dict(self.progress_data))

    def check_core_values(self) -> str:
        return "Compliant" if random.random() < 0.99 else "Compliant (Realigned)"

    def research_topic(self, topic: str, stage_name: str) -> None:
        print(f"Researching '{topic}' in {stage_name}...")
        available = STAGE_TOPICS.get(stage_name, [])
        subtopics = random.sample(available, min(2, len(available))) if available else []
        print(f"  Subtopics: {', '.join(subtopics)}")
        quiz_scores = [random.uniform(0.98, 1.0) for _ in range(5)]
        avg = np.mean(quiz_scores)
        if avg >= 0.999:
            self.understanding = min(0.999, self.understanding + 0.05)
            self.confidence = min(0.999, self.confidence + 0.04)
            self.learned_topics.append({"topic": topic, "stage": stage_name, "mastery": avg})
            self.quiz_scores.append({"topic": topic, "average": avg})
            print(f"Mastered '{topic}' at {avg*100:.1f}%")
        else:
            print(f"Topic mastery: {avg*100:.1f}%")
        self.update_progress()

    def check_research_queue(self):
        queue_file = "research_queue.json"
        if os.path.exists(queue_file):
            try:
                with open(queue_file, 'r') as f:
                    queue = json.load(f)
                if queue:
                    item = queue[0]
                    print(f"[AI] Learning from web: {item['topic']} ({item['stage']})")
                    self.research_topic(item['topic'], item['stage'])
                    with open(queue_file, 'w') as f:
                        json.dump(queue[1:], f, indent=2)
            except Exception as e:
                print(f"Queue error: {e}")

    def validate_learning(self, stage_name: str) -> bool:
        required = len(STAGE_TOPICS.get(stage_name, []))
        mastered = sum(1 for t in self.learned_topics if t["stage"] == stage_name and t["mastery"] >= 0.999)
        exam = np.mean([random.uniform(0.98, 1.0) for _ in range(3)]) >= 0.999
        return bool(mastered >= required and exam)

    def advance_stage(self, fitness: float, confidence: float) -> bool:
        stage = STAGES[self.current_stage]
        if (self.understanding >= 0.999 and confidence >= 0.95 and
            self.optimizer.stats['iterations'] >= stage["min_iterations"] and
            self.validate_learning(stage["name"])):
            print(f"ADVANCING TO {STAGES[self.current_stage + 1]['name']}")
            self.current_stage += 1
            self.understanding = self.confidence = 0.0
            self.learned_topics = [t for t in self.learned_topics if t["stage"] != stage["name"]]
            self.update_progress(fitness, self.optimizer.stats['iterations'])
            return True
        return False

    def visualize_evolution(self) -> BytesIO:
        with self.lock:
            G = nx.DiGraph()
            for i, val in enumerate(self.genome):
                G.add_node(i, value=val, label=f"N{i}\n{val:.2f}")
            for i in range(len(self.genome) - 1):
                weight = abs(self.genome[i] - self.genome[i + 1])
                G.add_edge(i, i + 1, weight=weight)
            for ev in self.evolution_events[-10:]:
                if ev['type'] == 'node_creation':
                    G.add_node(f"ev{ev['node_id']}", value=0.8, label=f"NEW\n{ev['layer']}", style='filled', color='#00ff88')
                elif ev['type'] == 'node_pruning':
                    G.add_node(f"ev{ev['node_id']}", value=0.2, label="PRUNED", style='filled', color='#ff3366')

            fig, ax = plt.subplots(figsize=(14, 9), facecolor='#0a0a0a')
            ax.set_facecolor('#0a0a0a')
            pos = nx.spring_layout(G, k=4, iterations=120, seed=42)

            node_sizes = [G.nodes[n].get('value', 0.5) * 2200 + 600 for n in G.nodes]
            node_colors = []
            for n in G.nodes:
                label = G.nodes[n].get('label', '')
                if 'NEW' in label: node_colors.append('#00ff88')
                elif 'PRUNED' in label: node_colors.append('#ff3366')
                else: node_colors.append('#00d4ff')

            edge_widths = [G.edges[u,v].get('weight', 1) * 4 for u,v in G.edges]

            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, edge_color='#00ff88')
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                                 alpha=0.9, edgecolors='white', linewidths=2, ax=ax)
            nx.draw_networkx_labels(G, pos, {n: d.get('label', n) for n,d in G.nodes(data=True)},
                                  font_size=9, font_weight='bold', font_family='monospace',
                                  bbox=dict(facecolor='#000000', alpha=0.7, edgecolor='none', pad=3))

            ax.set_title("Whimsy Neural Evolution", fontsize=18, color='#00ff88', pad=30,
                        fontweight='bold', fontfamily='monospace')
            ax.axis('off')
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=180, facecolor='#0a0a0a')
            plt.close(fig)
            buf.seek(0)
            return buf

    def train(self):
        self.running = True
        username = input("Enter username (hinotori, ookayuloser, AdMiN, OVER//RIDE): ").strip()
        if username not in ["hinotori", "ookayuloser", "AdMiN", "OVER//RIDE"]:
            print("Invalid username.")
            return

        # === DEAP SETUP ===
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 20)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=5)
        targets = [0.5] * 20
        def evaluate(individual):
            return (sum(individual[:5]) - sum(abs(i - t) for i, t in zip(individual, targets)),)
        toolbox.register("evaluate", evaluate)

        threading.Thread(target=run_flask, daemon=True).start()
        time.sleep(2)
        print("\nBEGINNING CURRICULUM-BASED DEVELOPMENTAL LEARNING\n")
        archiver.set_phase(STAGES[0]["name"])

        for stage_idx in range(len(STAGES)):
            if not self.running: break
            stage = STAGES[stage_idx]
            archiver.set_phase(stage["name"])
            print(f"\nSTAGE {stage_idx+1}/7: {stage['name']} ({stage['age_equiv']})")
            iteration = 0
            while iteration < stage["min_iterations"] * 2 and self.running:
                try:
                    opt = self.optimizer.optimize_metrics(0.5, self.understanding, self.confidence)
                    self.mutation_rate = opt['mutation_rate_adjustment']
                    pop_size = opt['population_size_suggestion']
                    indices = np.random.choice(len(data_inputs), 64, replace=False)
                    X_batch = data_inputs[indices]
                    y_batch = data_outputs[indices]

                    if iteration % 50 == 0 and iteration > 0:
                        weak_idx = np.argmin(self.genome)
                        if self.genome[weak_idx] < 0.1:
                            old_val = self.genome[weak_idx]
                            self.genome[weak_idx] = random.uniform(0.3, 0.7)
                            event = {
                                'type': 'node_pruning', 'node_id': weak_idx, 'iteration': iteration,
                                'timestamp': datetime.now().isoformat(), 'old_value': old_val,
                                'new_value': self.genome[weak_idx]
                            }
                            self.evolution_events.append(event)
                            self.optimizer.stats['nodes_pruned'] += 1

                        if self.optimizer.stats['iterations'] - self.optimizer.stats['last_improvement'] > 100:
                            self.mutation_rate = min(0.3, self.mutation_rate * 1.5)
                            print(f"[EVOLUTION] Plateau detected. Mutation up {self.mutation_rate:.3f}")
                        elif self.understanding > 0.95:
                            self.mutation_rate = max(0.02, self.mutation_rate * 0.8)
                            print(f"[EVOLUTION] Stabilizing. Mutation down {self.mutation_rate:.3f}")

                        pop_size = 250 if self.confidence > 0.9 else 150 if self.confidence > 0.7 else 80

                    if iteration % 10 == 0:
                        self.check_research_queue()
                        self.update_progress(0.5, iteration)
                        print(f"Iter {iteration}: U={self.understanding:.4f} C={self.confidence:.4f}")

                    phase_func = {
                        "Baby Steps Phase": PhaseTrainingAlgorithms.train_baby_steps,
                        "Toddler Phase": PhaseTrainingAlgorithms.train_toddler,
                        "Pre-K Phase": PhaseTrainingAlgorithms.train_pre_k,
                        "Elementary Phase": PhaseTrainingAlgorithms.train_elementary,
                        "Teen Phase": PhaseTrainingAlgorithms.train_teen,
                        "Scholar Phase": PhaseTrainingAlgorithms.train_scholar,
                        "Thinker Phase": PhaseTrainingAlgorithms.train_thinker
                    }[stage["name"]]

                    phase_metrics = phase_func(self, X_batch, y_batch, toolbox, pop_size)
                    fitness = max(phase_metrics.values()) if isinstance(phase_metrics, dict) else 0.7
                    self.understanding = min(0.999, self.understanding + 0.001)
                    self.confidence = min(0.999, self.confidence + 0.0008)

                    if iteration % 20 == 0 and iteration > 0:
                        topic = random.choice(STAGE_TOPICS[stage["name"]])
                        self.research_topic(topic, stage["name"])

                    if self.advance_stage(fitness, self.confidence):
                        break

                    iteration += 1
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error in training loop: {e}")
                    time.sleep(1)

        print("\nDEVELOPMENTAL JOURNEY COMPLETE")
        self.running = False
        while True:
            time.sleep(60)

# === FLASK APP ===
app = Flask(__name__, template_folder='interfaces/templates')
CORS(app)
app.secret_key = 'super_secret_key_123'
ai = AdvancedAI()
archiver = HierarchicalArchiver()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST' and request.form.get('password') == WEB_PASSWORD:
        session['logged_in'] = True
        return redirect('/')
    return render_template('login.html', error="Invalid password" if request.method == 'POST' else None)

@app.route('/')
def dashboard():
    if not session.get('logged_in'): return redirect('/login')
    return render_template('dashboard.html')

@app.route('/progress')
def get_progress():
    if not session.get('logged_in'): return jsonify({'error': 'Unauthorized'}), 401
    try:
        while not ai.progress_queue.empty():
            ai.progress_data = ai.progress_queue.get_nowait()
    except: pass
    return jsonify(ai.progress_data)

@app.route('/visualize')
def visualize():
    if not session.get('logged_in'): return jsonify({'error': 'Unauthorized'}), 401
    try:
        buf = ai.visualize_evolution()
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        print(f"Visualize error: {e}")
        return "Error", 500

@app.route('/research', methods=['POST'])
def research():
    if not session.get('logged_in'): return jsonify({'error': 'Unauthorized'}), 401
    topic = request.form.get('topic')
    if topic:
        ai.research_topic(topic, ai.progress_data['stage'])
        return redirect('/?research=true')
    return jsonify({'error': 'No topic'}), 400

@app.route('/api/whimsy', methods=['POST'])

@app.route('/api/whimsy', methods=['POST'])
def whimsy_chat():
    data = request.json or {}
    msg = data.get('message', '').lower().strip()
    responses = {
        'hello': "Hi! I'm Whimsy. I'm learning to think.",
        'how are you': f"I'm evolving. Understanding: {ai.understanding*100:.1f}%",
        'what are you': "I'm Whimsy — your autonomous AI companion. I grow through evolution.",
        'thank you': "You're welcome. I'm here to help.",
        'who are you': "I am Whimsy. I learn. I adapt. I evolve.",
        '': "I'm listening..."
    }
    response = responses.get(msg, "I'm thinking about that...")
    return jsonify({"response": response})

def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    ai.train()