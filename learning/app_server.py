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

    data = request.json or {}
    message = data.get("message", "").strip().lower()

    if not trainer:
        return jsonify({"response": "System initializing..."})

    stage = trainer.get_current_stage()['name']
    
    if message.startswith("evolve") or message.startswith("mutate"):
        trainer.autonomous_evolve()
        trainer.genetic_evolution()
        
        response = "🧬 EVOLUTION TRIGGERED!\n\n"
        response += f"📈 Generation: {trainer.generation}\n"
        response += f"🔬 Genetic patterns: {len(trainer.genetic_patterns)}\n"
        response += f"⚡ Evolution events: {len(trainer.evolution_events)}\n"
        
        if trainer.evolution_events:
            last_event = trainer.evolution_events[-1]
            response += f"\n🔄 Last mutation: {last_event.get('change', 'N/A')}\n"
            response += f"💡 Reason: {last_event.get('reason', 'N/A')}"
        
        return jsonify({
            "response": response,
            "stage": stage,
            "understanding": trainer.understanding,
            "is_evolution": True
        })
    
    elif message.startswith("status"):
        state = trainer.get_state()
        response = f"📊 WHIMSY STATUS REPORT\n\n"
        response += f"🎭 Stage: {state['stage']} ({state['stage_index'] + 1}/7)\n"
        response += f"📈 Generation: {state['generation']}\n"
        response += f"🔄 Iteration: {state['iteration']}\n"
        response += f"🧠 Understanding: {state['understanding']*100:.1f}%\n"
        response += f"🎯 Target: {state['target_understanding']*100:.1f}%\n"
        response += f"✅ Accuracy: {state['accuracy']*100:.1f}%\n"
        response += f"💪 Confidence: {state['confidence']*100:.1f}%\n"
        response += f"📚 Knowledge items: {state['knowledge_count']}\n"
        response += f"🧬 Evolution events: {state['evolution_count']}\n"
        response += f"🔧 Code rewrites: {len(trainer.code_rewrites)}\n"
        response += f"🧪 Genetic patterns: {len(trainer.genetic_patterns)}"
        
        return jsonify({
            "response": response,
            "stage": stage,
            "understanding": trainer.understanding,
            "is_status": True
        })
    
    elif message.startswith("rewrite") or message.startswith("self-modify"):
        trainer.rewrite_own_code()
        response = "🔧 CODE SELF-MODIFICATION TRIGGERED!\n\n"
        
        if trainer.code_rewrites:
            response += f"📝 Total rewrites: {len(trainer.code_rewrites)}\n\n"
            for rewrite in trainer.code_rewrites[-3:]:
                response += f"✏️ {rewrite.get('description', 'N/A')}\n"
                response += f"   File: {rewrite.get('file', 'N/A')}\n\n"
        else:
            response += "No code modifications applied yet. System analyzing performance..."
        
        return jsonify({
            "response": response,
            "stage": stage,
            "understanding": trainer.understanding,
            "is_rewrite": True
        })
    
    elif message.startswith("advance") or message.startswith("next stage"):
        current_stage = trainer.get_current_stage()
        if trainer.understanding >= current_stage['target_understanding']:
            old_stage = current_stage['name']
            if trainer.advance_stage():
                new_stage = trainer.get_current_stage()['name']
                response = f"🎉 STAGE ADVANCEMENT APPROVED!\n\n"
                response += f"📈 {old_stage} → {new_stage}\n"
                response += f"🎯 New target: {trainer.get_current_stage()['target_understanding']*100:.0f}%\n"
                response += f"📊 Current understanding reset to work toward new goal"
                return jsonify({
                    "response": response,
                    "stage": new_stage,
                    "understanding": trainer.understanding,
                    "is_advancement": True
                })
            else:
                response = "🏆 You've already reached the final stage: Thinker!"
        else:
            remaining = current_stage['target_understanding'] - trainer.understanding
            response = f"⏳ Not ready to advance yet.\n\n"
            response += f"📊 Current understanding: {trainer.understanding*100:.1f}%\n"
            response += f"🎯 Target: {current_stage['target_understanding']*100:.0f}%\n"
            response += f"📈 Need {remaining*100:.1f}% more to advance"
        
        return jsonify({
            "response": response,
            "stage": stage,
            "understanding": trainer.understanding,
            "is_advancement": False
        })
    
    elif message.startswith("genetic") or message.startswith("dna"):
        trainer.genetic_evolution()
        
        best_strategy = trainer.genetic_learner.get_best_strategy()
        patterns = trainer.genetic_learner.get_adaptive_patterns_summary()
        
        response = "🧬 GENETIC ALGORITHM STATUS\n\n"
        response += f"🔬 Generation: {trainer.genetic_learner.generation}\n"
        response += f"📊 Population: {trainer.genetic_learner.population_size} genomes\n"
        
        if best_strategy:
            response += f"\n🏆 Best Strategy:\n"
            response += f"   Learning rate: {best_strategy.get('learning_rate', 0):.6f}\n"
            response += f"   Momentum: {best_strategy.get('momentum', 0):.3f}\n"
            response += f"   Exploration: {best_strategy.get('exploration_rate', 0):.3f}\n"
            response += f"   Fitness: {best_strategy.get('fitness', 0):.4f}\n"
        
        if patterns.get('recent_patterns'):
            response += f"\n🔍 Recent Patterns Discovered:\n"
            for p in patterns['recent_patterns'][-3:]:
                response += f"   • {p['type']}: {p['description'][:50]}...\n"
        
        return jsonify({
            "response": response,
            "stage": stage,
            "understanding": trainer.understanding,
            "is_genetic": True
        })
    
    elif message.startswith("knowledge") or message.startswith("what do you know"):
        response = "📚 KNOWLEDGE BASE\n\n"
        response += f"Total learned: {len(trainer.knowledge)} topics\n\n"
        
        if trainer.knowledge:
            response += "Recent learning:\n"
            for k in trainer.knowledge[-5:]:
                topic = k.get('topic', 'Unknown')
                sources = k.get('sources', [])
                confidence = k.get('confidence', k.get('understanding', 0))
                response += f"\n📖 {topic}\n"
                response += f"   Sources: {', '.join(sources) if sources else 'chat-based'}\n"
                response += f"   Confidence: {confidence*100:.1f}%\n"
        else:
            response += "No knowledge acquired yet. Ask me to 'learn [topic]' to start!"
        
        return jsonify({
            "response": response,
            "stage": stage,
            "understanding": trainer.understanding,
            "is_knowledge": True
        })
    
    elif message.startswith("help") or message == "?":
        response = "🤖 WHIMSY COMMANDS\n\n"
        response += "📚 learn [topic] - Learn about a topic from web sources\n"
        response += "📊 status - Show current training status\n"
        response += "🎯 advance - Approve stage advancement (when ready)\n"
        response += "🧬 evolve - Trigger evolution cycle\n"
        response += "🧪 genetic - Show genetic algorithm status\n"
        response += "🔧 rewrite - Trigger code self-modification\n"
        response += "📖 knowledge - Show what I've learned\n"
        response += "❓ help - Show this help message\n\n"
        response += f"Current stage: {stage}\n"
        response += f"Target: {trainer.get_current_stage()['target_understanding']*100:.0f}%"
        
        return jsonify({
            "response": response,
            "stage": stage,
            "understanding": trainer.understanding,
            "is_help": True
        })
    
    learn_keywords = ["learn", "teach", "study", "understand", "research"]
    is_learning_request = any(kw in message for kw in learn_keywords)
    
    if is_learning_request:
        topic = message
        for kw in learn_keywords:
            topic = topic.replace(kw, "").strip()
        
        topic = topic.strip("about ").strip()
        
        if topic:
            learning_result = trainer.learn_topic(topic)
            
            understanding_pct = int(learning_result['understanding'] * 100)
            topic_understanding_pct = int(learning_result.get('topic_understanding', 0) * 100)
            
            response = f"🧠 LEARNING: '{topic.upper()}'\n"
            response += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            response += f"📊 Topic Understanding: {topic_understanding_pct}%\n"
            response += f"📈 Overall Understanding: {understanding_pct}%\n"
            response += f"🎯 Target: 99%\n\n"
            
            sources = learning_result.get('sources', [])
            if sources:
                response += f"🌐 Sources searched:\n"
                for src in sources:
                    response += f"   • {src.title()}\n"
                response += "\n"
            
            response += f"📚 Method: {learning_result.get('method', 'N/A')}\n"
            response += f"📝 Knowledge items: {learning_result.get('knowledge_items', 0)}\n\n"
            
            synthesized = learning_result.get('synthesized', '')
            if synthesized:
                response += f"💡 Synthesis:\n{synthesized[:300]}...\n\n"
            
            if learning_result.get('learning_complete'):
                response += f"✅ MASTERED! (99% understanding achieved)\n"
            else:
                remaining = 99 - understanding_pct
                response += f"⏳ Need {remaining}% more to reach mastery\n"
            
            if learning_result['stage_advanced']:
                response += f"\n🎉 STAGE ADVANCEMENT → {learning_result['stage']}!"
            
            return jsonify({
                "response": response,
                "stage": stage,
                "understanding": float(learning_result['understanding']),
                "topic_understanding": float(learning_result.get('topic_understanding', 0)),
                "topic": topic,
                "is_learning": True,
                "knowledge_items": int(learning_result['knowledge_items']),
                "sources": sources,
                "learning_complete": bool(learning_result.get('learning_complete', False))
            })
    
    responses = {
        "Baby Steps": f"(Baby Steps) goo... ba ba... type 'help' for commands!",
        "Toddler": f"(Toddler) me learning! Type 'help' to see what I can do!",
        "Pre-K": f"(Pre-K) I'm thinking... Type 'help' for available commands!",
        "Elementary": f"(Elementary) I'm curious! Type 'help' to see my abilities!",
        "Teen": f"(Teen) Ready to learn and evolve. Type 'help' for commands!",
        "Scholar": f"(Scholar) Ready for deep learning. Type 'help' for my capabilities!",
        "Thinker": f"(Thinker) Let's explore knowledge together. Type 'help' to begin!"
    }

    return jsonify({
        "response": responses.get(stage, "Type 'help' for commands!"),
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
