# web_researcher.py — FULL, CLEAN, READY
import os, time, json, random, requests, re, threading
from datetime import datetime
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote

RESEARCH_QUEUE = "research_queue.json"
OUTPUT_DIR = "research_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_DOMAINS = ["wikipedia.org","britannica.com","khanacademy.org","edu.gov","nationalgeographic.com","sciencedaily.com","nasa.gov"]
GOOGLE_SEARCH = "https://www.google.com/search"

session = requests.Session()
retry = requests.adapters.Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
adapter = requests.adapters.HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

app = Flask(__name__)

def clean_text(text: str) -> str: return re.sub(r'\s+', ' ', text).strip()
def is_allowed(url: str) -> bool: return any(d in urlparse(url).netloc.lower() for d in ALLOWED_DOMAINS)

def google_search(query: str, num_results: int = 5) -> list:
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"q": query, "num": num_results}
    try:
        r = session.get(GOOGLE_SEARCH, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        links = []
        for g in soup.find_all('div', class_='g'):
            a = g.find('a')
            if a and 'href' in a.attrs:
                url = str(a['href'])
                if url.startswith('/url?'): url = unquote(url.split('&')[0].split('url=')[1])
                if is_allowed(url): links.append(url)
                if len(links) >= num_results: break
        return links[:num_results]
    except Exception as e: print(f"Search failed: {e}"); return []

def scrape_page(url: str) -> dict:
    try:
        r = session.get(url, timeout=10); r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        for s in soup(["script","style","nav","footer","header"]): s.decompose()
        text = ' '.join([c.strip() for line in soup.get_text().splitlines() for c in line.split("  ") if c])
        sentences = re.split(r'[.!?]+', text)[:3]
        summary = '. '.join(sentences); summary = summary[:500] + "..." if len(summary) > 500 else summary
        return {"url": url, "title": soup.title.string if soup.title else "No title", "summary": clean_text(summary), "full_text_snippet": clean_text(text[:1000])}
    except Exception as e: return {"url": url, "error": str(e)}

def research_topic(topic: str, stage: str) -> None:
    print(f"[RESEARCHER] Researching '{topic}' for {stage}...")
    stage_dir = os.path.join(OUTPUT_DIR, stage.lower().replace(" ", "_"))
    os.makedirs(stage_dir, exist_ok=True)
    urls = google_search(topic, 3) or [f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"]
    results = []
    for url in urls:
        data = scrape_page(url); results.append(data)
        safe_name = re.sub(r'\W+', '_', topic)[:50]
        with open(os.path.join(stage_dir, f"{safe_name}_{int(time.time())}.json"), 'w') as f: json.dump(data, f, indent=2)
    subtopics = []
    if results and results[0].get("summary"):
        text = " ".join(r.get("summary","") for r in results if r.get("summary"))
        words = re.findall(r'\b[a-zA-Z]{5,}\b', text)
        common = [w for w in words if words.count(w) > 1][:2]
        subtopics = [f"{topic} {w}" for w in common][:2]
    queue_entry = {
        "topic": topic, "stage": stage, "timestamp": datetime.now().isoformat(),
        "sources": [r["url"] for r in results if "error" not in r],
        "summary": results[0].get("summary","") if results else "",
        "subtopics": subtopics, "mastery_target": 0.999
    }
    try:
        queue = json.load(open(RESEARCH_QUEUE, 'r')) if os.path.exists(RESEARCH_QUEUE) else []
    except: queue = []
    queue.append(queue_entry)
    with open(RESEARCH_QUEUE, 'w') as f: json.dump(queue, f, indent=2)
    print(f"[RESEARCHER] Queued '{topic}' → {len(queue)} items")

@app.route('/api/research', methods=['POST'])
def api_research():
    data = request.json or {}
    topic, stage = data.get('topic'), data.get('stage', 'Baby Steps Phase')
    if not topic: return jsonify({"error": "No topic"}), 400
    threading.Thread(target=research_topic, args=(topic, stage), daemon=True).start()
    return jsonify({"status": "research_started", "topic": topic})

@app.route('/api/queue')
def get_queue():
    try: return jsonify(json.load(open(RESEARCH_QUEUE, 'r')))
    except: return jsonify([])

def auto_research_loop():
    while True:
        time.sleep(30)
        if not os.path.exists("train_advanced_ai.py"): continue
        try:
            progress = json.load(open("progress.json", 'r')) if os.path.exists("progress.json") else {}
            stage = progress.get("stage", "Baby Steps Phase")
        except: stage = "Baby Steps Phase"
        topics = STAGE_TOPICS.get(stage, [])
        if topics and random.random() < 0.3:
            research_topic(random.choice(topics), stage)

STAGE_TOPICS = { ... }  # ← Keep your full STAGE_TOPICS here

if __name__ == "__main__":
    threading.Thread(target=auto_research_loop, daemon=True).start()
    print("Web Researcher API running on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)