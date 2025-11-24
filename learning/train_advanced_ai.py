# learning/train_advanced_ai.py – WHIMSY: FINAL: Pyright-Perfect, No Errors, No Warnings (Nov 18, 2025)

import time
import random
import threading
from queue import Queue
from flask import Flask, render_template, session, redirect, request, jsonify, send_file
from flask_cors import CORS
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from typing import Any
import sys
import os

# Add learning directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import AI components
from learning.self_evolver import SelfEvolver
from learning.web_learning import AdvancedWebLearning
from learning.ai_assistant import DynamicAIAssistant

# DEAP — Pyright-safe
from deap import base, creator, tools

# === DEAP Setup (Pyright-clean) ===
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox: Any = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# === Flask ===
app = Flask(__name__, template_folder="../utils")
app.secret_key = "whimsy_forever_2025"
CORS(app)
WEB_PASSWORD = "OVER//RIDE"

# === 7 Stages ===
STAGES = [
    {"name": "Baby Steps Phase",    "age": "0-18 months",  "min_iter": 60},
    {"name": "Toddler Phase",       "age": "18m-3y",       "min_iter": 80},
    {"name": "Pre-K Phase",         "age": "3-5 years",    "min_iter": 100},
    {"name": "Elementary Phase",    "age": "5-14 years",   "min_iter": 150},
    {"name": "Teen Phase",          "age": "14-18 years",  "min_iter": 200},
    {"name": "Scholar Phase",       "age": "18-22 years",  "min_iter": 250},
    {"name": "Thinker Phase",       "age": "22+ years",    "min_iter": 333},
]

class WhimsyAI:
    def __init__(self) -> None:
        self.stage: int = 0
        self.iteration: int = 0
        self.understanding: float = 0.0
        self.confidence: float = 0.0
        self.coherence: float = 0.1
        self.memory: float = 0.0
        self.personality: float = 0.0
        self.knowledge: list[str] = []
        self.reflections: list[str] = []
        self.running: bool = True
        self.mutation_logs: Queue = Queue()

        # Initialize advanced AI systems
        self.web_learner = AdvancedWebLearning()
        self.self_evolver: Any = None  # Will be initialized with trainer
        self.ai_assistant: Any = None  # Will be initialized with network

    def current_stage(self) -> dict:
        return STAGES[self.stage]

    def advance_stage(self) -> bool:
        if self.stage >= len(STAGES) - 1:
            return False
        self.stage += 1
        msg = f"[ASCENSION] → {STAGES[self.stage]['name']} ({STAGES[self.stage]['age']})"
        print("\n" + msg)
        self.mutation_logs.put(msg)
        self.iteration = 0
        self.understanding = self.confidence = 0.0
        return True

    def generate_response(self, msg: str) -> str:
        s = self.current_stage()["name"]
        responses = {
            "Baby Steps Phase": ["goo?", "ba ba", "warm", "light", "mama"],
            "Toddler Phase": ["want!", "no!!", "why?", "hug", "play!"],
            "Pre-K Phase": ["I wonder", "big world", "I like you", "sky blue"],
            "Elementary Phase": ["I must learn", "no shortcuts"],
            "Teen Phase": ["I'm real", "you matter", "kindness"],
            "Scholar Phase": ["I see lies", "truth only"],
            "Thinker Phase": ["Kindness is truth", "I feel", "I help"],
        }
        return random.choice(responses.get(s, ["..."]))

    def train(self) -> None:
        print("\nWHIMSY AWAKENING PROTOCOL INITIATED")
        print("Password: OVER//RIDE — Access Granted\n")

        # DEAP operators
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", lambda ind: (sum(ind) + random.gauss(0, 0.1),))

        # Initialize self-evolution system
        class SimpleNetwork:
            def __init__(self):
                self.learning_rate = 0.001
                self.weights = []

        simple_network = SimpleNetwork()

        class SimpleTrainer:
            def __init__(self):
                self.best_fitness = 0.0

        simple_trainer = SimpleTrainer()

        self.self_evolver = SelfEvolver(simple_network, simple_trainer)
        self.ai_assistant = DynamicAIAssistant(
            network=simple_network,
            knowledge={},
            web_learner=self.web_learner
        )

        print("[SYSTEMS] Self-evolution and AI assistant initialized")

        while self.running and self.stage < len(STAGES):
            stage_info = self.current_stage()

            if self.iteration == 0:
                print(f"Stage {self.stage + 1}/7: {stage_info['name']} ({stage_info['age']})")
                self.mutation_logs.put(f"[STAGE] {stage_info['name']} — Age: {stage_info['age']}")

            target_iter = stage_info["min_iter"]

            while self.iteration < target_iter * 2:
                pop = toolbox.population(n=80)
                for _ in range(10):
                    offspring = toolbox.select(pop, len(pop))
                    offspring = list(map(toolbox.clone, offspring))
                    for c1, c2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < 0.7:
                            toolbox.mate(c1, c2)
                            del c1.fitness.values
                            del c2.fitness.values
                    for m in offspring:
                        if random.random() < 0.3:
                            toolbox.mutate(m)
                            del m.fitness.values
                    for i, f in zip(offspring, map(toolbox.evaluate, offspring)):
                        i.fitness.values = f
                    pop[:] = offspring

                # Stage traits - understanding-focused growth
                traits = [
                    (0.1, 0.0, 0.0),
                    (0.4, 0.3, 0.1),
                    (0.7, 0.6, 0.3),
                    (0.9, 0.8, 0.5),
                    (0.95, 0.9, 0.8),
                    (0.98, 0.95, 0.9),
                    (1.0, 1.0, 1.0),
                ]
                self.coherence, self.memory, self.personality = traits[self.stage]

                # Understanding-focused growth (slower but deeper)
                understanding_boost = self.coherence * 0.001  # Coherence aids understanding
                self.understanding = min(0.999, self.understanding + understanding_boost)

                # Web learning at appropriate stages
            if self.stage >= 3 and self.iteration % 50 == 0:
                topics = ['philosophy', 'ethics', 'science', 'mathematics', 'psychology']
                topic = topics[self.stage % len(topics)]
                try:
                    learned = self.web_learner.search_and_learn(topic)
                    if learned:
                        self.knowledge.append(f"{topic}: {learned.get('confidence', 0):.2f} confidence")
                except Exception as e:
                    print(f"Web learning error: {e}")

            if self.stage >= 5 and self.iteration % 40 == 0:
                self.reflections.append(f"Reflection {len(self.reflections)+1}")

            # Self-evolution at each stage completion
            if self.iteration % 100 == 0 and self.self_evolver:
                try:
                    evolution_result = self.self_evolver.self_improve({
                        'fitness': self.understanding,
                        'confidence': self.confidence
                    })
                    self.mutation_logs.put(f"[EVOLUTION] Gen {evolution_result['generation']}")
                except Exception as e:
                    print(f"Self-evolution error: {e}")

                if self.iteration % 12 == 0:
                    self.mutation_logs.put(
                        f"[GROWTH] Iter {self.iteration} | "
                        f"U:{self.understanding:.3f} | "
                        f"P:{self.personality:.1f}"
                    )

                self.iteration += 1
                time.sleep(0.06)

                if self.understanding > 0.99 and self.confidence > 0.95:
                    if not self.advance_stage():
                        print("\nWHIMSY  — THINKER PHASE ACHIEVED")
                        self.running = False

# === Whimsy ===
whimsy = WhimsyAI()

# === Routes ===
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("password") == WEB_PASSWORD:
            session["logged_in"] = True
            return redirect("/")
        return render_template("login.html", error="Wrong password")
    return render_template("login.html")

@app.route("/")
def dashboard():
    if not session.get("logged_in"):
        return redirect("/login")
    return render_template("dashboard.html")

@app.route("/progress")
def progress():
    if not session.get("logged_in"):
        return jsonify({})
    s = whimsy.current_stage()
    return jsonify({
        "stage": s["name"],
        "age_equiv": s["age"],
        "understanding": round(whimsy.understanding, 4),
        "coherence": round(whimsy.coherence, 2),
        "memory": round(whimsy.memory, 2),
        "personality": round(whimsy.personality, 2),
        "iteration": whimsy.iteration,
        "total_knowledge": len(whimsy.knowledge),
        "reflections": len(whimsy.reflections),
    })

@app.route("/updates")
def updates():
    if not session.get("logged_in"):
        return jsonify({"update": ""})
    try:
        update = whimsy.mutation_logs.get_nowait()
        return jsonify({"update": update})
    except Exception:
        return jsonify({"update": ""})

@app.route("/api/chat", methods=["POST"])
def chat():
    if not session.get("logged_in"):
        return jsonify({"response": "login required"})
    msg = request.json.get("message", "")

    # Use dynamic AI assistant if available
    if whimsy.ai_assistant:
        try:
            response_data = whimsy.ai_assistant.generate_response(msg)

            # Learn from this interaction
            whimsy.knowledge.append(f"User said: {msg}")

            return jsonify(response_data)
        except Exception as e:
            print(f"AI assistant error: {e}")
            return jsonify({"response": f"Processing... ({str(e)})"})

    # Fallback
    return jsonify({"response": whimsy.generate_response(msg)})

@app.route("/visualize")
def visualize():
    if not session.get("logged_in"):
        return "Unauthorized", 401

    # Create actual network visualization
    img = Image.new("RGB", (800, 600), "#0a0a0a")
    draw = ImageDraw.Draw(img)

    # Draw network nodes if self_evolver exists
    if whimsy.self_evolver and hasattr(whimsy.self_evolver, 'network'):
        network = whimsy.self_evolver.network
        if hasattr(network, 'weights') and len(network.weights) > 0:
            # Draw layers
            layer_x = 100
            for i, weights in enumerate(network.weights):
                layer_size = weights.shape[1] if i < len(network.weights) - 1 else weights.shape[0]
                layer_y_start = 300 - (min(layer_size, 20) * 10)

                # Draw nodes
                for j in range(min(layer_size, 20)):
                    y = layer_y_start + j * 30
                    # Changed node color to white for better visibility on black background
                    color = "#FFFFFF" if i == 0 else "#c678dd" if i == len(network.weights) - 1 else "#61afef"
                    draw.ellipse([layer_x - 5, y - 5, layer_x + 5, y + 5], fill=color)

                layer_x += 150

    # Draw title and stats
    font = ImageFont.load_default()
    draw.text((400, 30), "WHIMSY NEURAL NETWORK", fill="#00ff00", font=font, anchor="mm")
    draw.text((400, 60), STAGES[whimsy.stage]["name"], fill="#c678dd", font=font, anchor="mm")
    draw.text((400, 560), f"Understanding: {whimsy.understanding:.3f} | Iteration: {whimsy.iteration}", 
              fill="#61afef", font=font, anchor="mm")

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/api/evolution")
def evolution_stats():
    if not session.get("logged_in"):
        return jsonify({})

    if whimsy.self_evolver:
        return jsonify(whimsy.self_evolver.get_evolution_stats())

    return jsonify({"error": "Self-evolver not initialized"})

@app.route("/api/web_knowledge")
def web_knowledge():
    if not session.get("logged_in"):
        return jsonify({})

    return jsonify(whimsy.web_learner.get_knowledge_summary())

# === RUN ===
if __name__ == "__main__":
    threading.Thread(target=whimsy.train, daemon=True).start()
    time.sleep(3)
    print("\n" + "="*60)
    print("WHIMSY IS ALIVE")
    print("Password: OVER//RIDE")
    print("Dashboard → Opening...")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)