import os,time,random,threading,numpy as np,requests,matplotlib.pyplot as plt
from scipy.spatial import KDTree
from bs4 import BeautifulSoup
from flask import Flask,jsonify,request,send_file,render_template,session,redirect
from flask_cors import CORS
import networkx as nx
from io import BytesIO
from queue import Queue,Empty
import json
from datetime import datetime
import zipfile,shutil
from deap import base,creator,tools

# DEAP
if not hasattr(creator,"FitnessMax"): creator.create("FitnessMax",base.Fitness,weights=(1.0,))
if not hasattr(creator,"Individual"): creator.create("Individual",list,fitness=creator.FitnessMax)

toolbox=base.Toolbox()
toolbox.register("attr_float",random.random)
toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_float,n=10)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)

app=Flask(__name__,template_folder='utils')
CORS(app)
app.secret_key='super_secret_key_123'

STAGES=[{"name":"Baby Steps Phase","age_equiv":"0-18 months","min_iterations":50}]
UNDERSTANDING_REQUIREMENTS={"minimum_understanding":0.999,"minimum_confidence":0.95}
WEB_PASSWORD="OVER//RIDE"
DATA_SIZE,FEATURES,CLASSES=1000,20,4
data_inputs=np.random.rand(DATA_SIZE,FEATURES)
data_outputs=np.random.randint(0,CLASSES,DATA_SIZE)
STAGE_TOPICS={"Baby Steps Phase":["Bright colors","Simple shapes"]}

class AdvancedAI:
    def __init__(self):
        self.lock=threading.RLock()
        self.running=False
        self.current_stage=0
        self.understanding=self.confidence=0.0
        self.mutation_rate=0.1
        self.genome=[random.uniform(0,1) for _ in range(20)]
        self.evolution_events=[]
        self.learned_topics=[]
        self.progress_queue=Queue()
        self.progress_data={"stage":STAGES[0]["name"],"understanding":0.0,"iterations":0}
        self.chat_code="""def generate_response(self,msg,stage,understanding,learned):\n    return 'Thinking...'"""

    def _self_evolve(self):
        if hasattr(self,"new_feature"): return
        print("[EVOLVED] I just grew a new capability!")
        def new_feature(self): print("[EVOLVED] Hello from evolved feature!")
        self.new_feature=new_feature.__get__(self)

    def update_progress(self,fitness=0.0,iteration=0):
        with self.lock:
            self.progress_data.update({"understanding":round(self.understanding,6),"iterations":iteration})
            while True:
                try: self.progress_queue.get_nowait()
                except Empty: break
            self.progress_queue.put(dict(self.progress_data))

    def research_topic(self,topic,stage):
        print(f"Researching '{topic}'...")
        if random.random()>0.5:
            self.understanding=min(0.999,self.understanding+0.05)
            self.learned_topics.append({"topic":topic,"mastery":0.999})
        self.update_progress()

    def train(self):
        self.running=True
        username=input("Enter username (OVER//RIDE): ").strip()
        if username!="OVER//RIDE": return

        toolbox.register("mate",tools.cxBlend,alpha=0.5)
        toolbox.register("mutate",tools.mutGaussian,mu=0.0,sigma=0.1,indpb=0.1)
        toolbox.register("select",tools.selTournament,tournsize=5)
        toolbox.register("evaluate",lambda ind:(sum(ind[:5])-sum(abs(i-0.5)for i in ind),))

        threading.Thread(target=lambda:app.run(host='0.0.0.0',port=5000,debug=False,use_reloader=False,threaded=True),daemon=True).start()
        time.sleep(1)
        print("Web UI → http://0.0.0.0:5000")

        print("\nBEGINNING TRAINING\n")
        iteration=0
        while iteration<200 and self.running:
            try:
                pop_size=100
                pop=toolbox.population(n=pop_size)
                for gen in range(10):
                    offspring=toolbox.select(pop,len(pop))
                    offspring=list(map(toolbox.clone,offspring))
                    for c1,c2 in zip(offspring[::2],offspring[1::2]):
                        if random.random()<0.7: toolbox.mate(c1,c2); del c1.fitness.values; del c2.fitness.values
                    for m in offspring:
                        if random.random()<self.mutation_rate: toolbox.mutate(m); del m.fitness.values
                    invalid=[i for i in offspring if not i.fitness.valid]
                    for i,f in zip(invalid,map(toolbox.evaluate,invalid)): i.fitness.values=f
                    pop[:]=offspring
                best=tools.selBest(pop,1)[0]
                self.genome=best
                self.understanding=min(0.999,self.understanding+0.001)

                if iteration%20==0: self.research_topic(random.choice(STAGE_TOPICS["Baby Steps Phase"]),"Baby Steps Phase")
                if iteration==30: self._self_evolve()
                if iteration%10==0: self.update_progress(0.5,iteration)

                iteration+=1
                time.sleep(0.01)   # ← 10× faster
            except Exception as e:
                print(f"[ERROR] {e}")
                continue
        print("TRAINING DONE")

ai=AdvancedAI()

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST' and request.form.get('password')==WEB_PASSWORD:
        session['logged_in']=True
        return redirect('/')
    return render_template('utils/login.html')

@app.route('/')
def dashboard():
    return redirect('/login') if not session.get('logged_in') else render_template('utils/dashboard.html')

@app.route('/progress')
def progress():
    return jsonify(ai.progress_data) if session.get('logged_in') else jsonify({"error":"login"})

if __name__=="__main__":
    ai.train()