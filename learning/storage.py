import json
import pickle
from datetime import datetime
import os

class KnowledgeStorage:
    """
    Store all learned information, solutions, and insights
    Tracks what has been learned and understood
    """
    
    def __init__(self, storage_dir='knowledge'):
        self.storage_dir = storage_dir
        self.learned_concepts = {}
        self.solutions = []
        self.insights = []
        self.understanding_records = []
        
        os.makedirs(storage_dir, exist_ok=True)
    
    def store_learned_concept(self, concept_id, concept_name, stage, understanding_score, details):
        """Store a concept that has been learned"""
        concept = {
            'id': concept_id,
            'name': concept_name,
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'understanding_score': understanding_score,
            'details': details,
            'mastered': understanding_score >= 0.999
        }
        
        self.learned_concepts[concept_id] = concept
        return concept
    
    def store_solution(self, problem, solution, method, accuracy, confidence):
        """Store a solution to a problem"""
        solution_entry = {
            'id': len(self.solutions),
            'timestamp': datetime.now().isoformat(),
            'problem': problem,
            'solution': solution,
            'method': method,
            'accuracy': accuracy,
            'confidence': confidence,
            'verified': accuracy >= 0.95 and confidence >= 0.99
        }
        
        self.solutions.append(solution_entry)
        return solution_entry['id']
    
    def store_insight(self, insight_text, category, stage, importance=5):
        """Store an insight or realization"""
        insight = {
            'id': len(self.insights),
            'timestamp': datetime.now().isoformat(),
            'insight': insight_text,
            'category': category,
            'stage': stage,
            'importance': importance
        }
        
        self.insights.append(insight)
        return insight['id']
    
    def record_understanding(self, stage, topic, metrics):
        """Record understanding achievement for a topic"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'topic': topic,
            'understanding_score': metrics.get('understanding', 0),
            'accuracy': metrics.get('raw_accuracy', 0),
            'confidence': metrics.get('confidence', 0),
            'consistency': metrics.get('consistency', 1.0),
            'achieved_threshold': metrics.get('understanding', 0) >= 0.999
        }
        
        self.understanding_records.append(record)
        return record
    
    def get_mastered_concepts(self):
        """Get all concepts that have been mastered (>99.9% understanding)"""
        return {
            concept_id: concept 
            for concept_id, concept in self.learned_concepts.items() 
            if concept['mastered']
        }
    
    def get_verified_solutions(self):
        """Get all verified solutions"""
        return [s for s in self.solutions if s['verified']]
    
    def get_knowledge_summary(self):
        """Get summary of all knowledge"""
        mastered = self.get_mastered_concepts()
        verified_solutions = self.get_verified_solutions()
        
        return {
            'total_concepts': len(self.learned_concepts),
            'mastered_concepts': len(mastered),
            'total_solutions': len(self.solutions),
            'verified_solutions': len(verified_solutions),
            'total_insights': len(self.insights),
            'understanding_records': len(self.understanding_records),
            'mastery_rate': len(mastered) / len(self.learned_concepts) if self.learned_concepts else 0
        }
    
    def save_solution_log(self, filepath='knowledge/solution_log.json'):
        """Save solution log in JSON format"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'learned_concepts': self.learned_concepts,
            'solutions': self.solutions,
            'insights': self.insights,
            'understanding_records': self.understanding_records,
            'summary': self.get_knowledge_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Solution log saved to {filepath}")
    
    def save_all_knowledge(self):
        """Save all knowledge to separate files"""
        self.save_solution_log()
        
        with open(f'{self.storage_dir}/learned_concepts.json', 'w') as f:
            json.dump(self.learned_concepts, f, indent=2)
        
        with open(f'{self.storage_dir}/insights.json', 'w') as f:
            json.dump(self.insights, f, indent=2)
        
        with open(f'{self.storage_dir}/understanding_records.json', 'w') as f:
            json.dump(self.understanding_records, f, indent=2)
    
    def load_solution_log(self, filepath='knowledge/solution_log.json'):
        """Load solution log from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.learned_concepts = data.get('learned_concepts', {})
            self.solutions = data.get('solutions', [])
            self.insights = data.get('insights', [])
            self.understanding_records = data.get('understanding_records', [])
            
            print(f"Solution log loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"No existing solution log found at {filepath}")
            return False
