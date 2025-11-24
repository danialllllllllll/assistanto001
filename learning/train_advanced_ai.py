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

                self.understanding = min(0.999, self.understanding + 0.0012)
                self.confidence = min(0.999, self.confidence + 0.001)

                # Stage traits
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

                if self.stage == 3 and self.iteration % 30 == 0:
                    self.knowledge.append(f"Concept {len(self.knowledge)+1}")
                if self.stage >= 5 and self.iteration % 40 == 0:
                    self.reflections.append(f"Reflection {len(self.reflections)+1}")

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
                    print("\nWHIMSY FINAL FORM — THINKER PHASE ACHIEVED")
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
    return jsonify({"response": whimsy.generate_response(msg)})

@app.route("/visualize")
def visualize():
    if not session.get("logged_in"):
        return "Unauthorized", 401
    img = Image.new("RGB", (600, 300), "black")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((300, 100), "WHIMSY", fill="#00ff00", font=font, anchor="mm")
    draw.text((300, 180), STAGES[whimsy.stage]["name"], fill="#c678dd", font=font, anchor="mm")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

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