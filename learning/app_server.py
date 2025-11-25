from flask import Flask, render_template, session, redirect, request, jsonify, send_file
from flask_cors import CORS
import threading
import time
import json
import os
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__, template_folder='../utils')
app.secret_key = os.urandom(24)
CORS(app)

WEB_PASSWORD = "OVER//RIDE"

trainer = None

def init_trainer():
    global trainer
    from learning.train_whimsy import trainer as t
    trainer = t
    threading.Thread(target=trainer.train_loop, daemon=True).start()
    print("Trainer initialized and started")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        password = request.form.get("password", "")
        if password == WEB_PASSWORD:
            session["logged_in"] = True
            session.permanent = True
            return redirect("/")
        return render_template("login.html", error="Incorrect password")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

@app.route("/")
def dashboard():
    if not session.get("logged_in"):
        return redirect("/login")
    return render_template("dashboard.html")

@app.route("/api/status")
def get_status():
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401

    if not trainer:
        return jsonify({"error": "Trainer not initialized"}), 503

    return jsonify(trainer.get_state())

@app.route("/api/updates")
def get_updates():
    if not session.get("logged_in"):
        return jsonify({"updates": []})

    if not trainer:
        return jsonify({"updates": []})

    updates = []
    while not trainer.update_queue.empty():
        try:
            updates.append(trainer.update_queue.get_nowait())
        except:
            break

    return jsonify({"updates": updates})

@app.route("/api/knowledge")
def get_knowledge():
    if not session.get("logged_in"):
        return jsonify({"knowledge": []})

    if not trainer:
        return jsonify({"knowledge": []})

    return jsonify({
        "knowledge": trainer.knowledge[-20:],
        "total": len(trainer.knowledge)
    })

@app.route("/api/evolution")
def get_evolution():
    if not session.get("logged_in"):
        return jsonify({"events": []})

    if not trainer:
        return jsonify({"events": []})

    return jsonify({
        "events": trainer.evolution_events[-20:],
        "total": len(trainer.evolution_events),
        "hyperparameters": trainer.evolver.hyperparameters
    })

@app.route("/api/network_viz")
def get_network_viz():
    if not session.get("logged_in"):
        return jsonify({"nodes": [], "connections": []})

    try:
        with open('data/network_topology.json', 'r') as f:
            return jsonify(json.load(f))
    except:
        return jsonify({"nodes": [], "connections": [], "stats": {}})

@app.route("/api/visualize")
def visualize_network():
    if not session.get("logged_in"):
        return "Unauthorized", 401

    img = Image.new("RGB", (800, 600), "#0a0a0a")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    if trainer:
        state = trainer.get_state()

        draw.text((400, 30), "WHIMSY NEURAL NETWORK", fill="#00ff00", font=font, anchor="mm")
        draw.text((400, 60), f"Stage: {state['stage']}", fill="#c678dd", font=font, anchor="mm")
        draw.text((400, 90), f"Gen: {state['generation']} | Iter: {state['iteration']}",
                  fill="#61afef", font=font, anchor="mm")

        layers = [32, 64, 32, 4]
        layer_x = 150

        for layer_idx, size in enumerate(layers):
            y_start = 300 - (min(size, 15) * 8)

            for node_idx in range(min(size, 15)):
                y = y_start + node_idx * 40
                activation = 0.5 + (state['understanding'] * 0.5)

                if activation > 0.7:
                    color = "#00ff00"
                elif activation > 0.4:
                    color = "#ffff00"
                else:
                    color = "#ff6600"

                draw.ellipse([layer_x - 6, y - 6, layer_x + 6, y + 6], fill=color)

            layer_x += 200

        draw.text((400, 560), f"Understanding: {state['understanding']:.3f} | Target: {state['target_understanding']:.2f}",
                  fill="#61afef", font=font, anchor="mm")
    else:
        draw.text((400, 300), "Initializing...", fill="#ffffff", font=font, anchor="mm")

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/api/realtime_viz")
def get_realtime_viz():
    if not session.get("logged_in"):
        return jsonify({"error": "Not logged in"}), 401

    if not trainer:
        return jsonify({"error": "Trainer not initialized"}), 503

    try:
        viz_data = trainer.realtime_viz.get_realtime_visualization_data()
        return jsonify(viz_data)
    except:
        return jsonify({"error": "Visualization data not available"}), 500

@app.route("/api/genetic_patterns")
def get_genetic_patterns():
    if not session.get("logged_in"):
        return jsonify({"patterns": []})

    if not trainer:
        return jsonify({"patterns": []})

    return jsonify({
        "patterns": trainer.genetic_patterns[-20:],
        "total": len(trainer.genetic_patterns),
        "best_strategy": trainer.genetic_learner.get_best_strategy(),
        "adaptive_patterns": trainer.genetic_learner.get_adaptive_patterns_summary()
    })

@app.route("/api/code_rewrites")
def get_code_rewrites():
    if not session.get("logged_in"):
        return jsonify({"rewrites": []})

    if not trainer:
        return jsonify({"rewrites": []})

    return jsonify({
        "rewrites": trainer.code_rewrites[-20:],
        "total": len(trainer.code_rewrites),
        "summary": trainer.code_rewriter.get_rewrite_summary() if trainer.code_rewriter else {}
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    if not session.get("logged_in"):
        return jsonify({"response": "Please log in first"})

    data = request.json
    message = data.get("message", "").strip()

    if not trainer:
        return jsonify({"response": "System initializing..."})

    # Check if user is asking to learn a topic
    learn_keywords = ["learn", "teach", "study", "understand", "research"]
    is_learning_request = any(kw in message.lower() for kw in learn_keywords)
    
    if is_learning_request:
        # Extract topic from message (simple approach: remove learn keywords)
        topic = message
        for kw in learn_keywords:
            topic = topic.lower().replace(kw, "").strip()
        
        if topic:
            # Use stage-specific algorithm to learn the topic
            learning_result = trainer.learn_topic(topic)
            
            stage = trainer.get_current_stage()['name']
            understanding_pct = int(learning_result['understanding'] * 100)
            
            response = f"🧠 Learning about '{topic}' at {stage} level...\n"
            response += f"📊 Understanding: {understanding_pct}%\n"
            response += f"🎯 Target: 99%\n"
            
            if learning_result['stage_advanced']:
                response += f"🎉 ADVANCED TO: {learning_result['stage']}!"
            
            return jsonify({
                "response": response,
                "stage": stage,
                "understanding": learning_result['understanding'],
                "topic": topic,
                "is_learning": True,
                "knowledge_items": learning_result['knowledge_items']
            })
    
    # Non-learning conversation
    stage = trainer.get_current_stage()['name']
    responses = {
        "Baby Steps": f"(Baby Steps) goo... ba ba...",
        "Toddler": f"(Toddler) me learning! Tell me what to learn!",
        "Pre-K": f"(Pre-K) I'm thinking... Tell me something to learn about!",
        "Elementary": f"(Elementary) I'm curious! What should I learn?",
        "Teen": f"(Teen) I'd like to expand my knowledge. What topic interests you?",
        "Scholar": f"(Scholar) I'm ready for deeper learning. What shall we study?",
        "Thinker": f"(Thinker) What profound subject shall we explore together?"
    }

    return jsonify({
        "response": responses.get(stage, "Tell me what to learn!"),
        "stage": stage,
        "understanding": trainer.understanding,
        "is_learning": False
    })

def start_server():
    init_trainer()
    time.sleep(2)
    print("\n" + "="*60)
    print("WHIMSY LEARNING SYSTEM")
    print("URL: http://0.0.0.0:5000")
    print("Password: OVER//RIDE")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    start_server()
