import json
from datetime import datetime

class NarrativeMemory:
    """
    Store experiences and reflections as narrative memories
    Creates coherent story of AI's development
    """
    
    def __init__(self):
        self.memories = []
        self.reflections = []
        self.milestones = []
        self.narrative_arc = []
    
    def add_memory(self, stage, event_type, description, emotional_context=None, importance=5):
        """
        Add a memory with emotional context
        importance: 1-10 scale
        """
        memory = {
            'id': len(self.memories),
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'event_type': event_type,
            'description': description,
            'emotional_context': emotional_context,
            'importance': importance
        }
        
        self.memories.append(memory)
        
        if importance >= 8:
            self.milestones.append(memory)
        
        return memory['id']
    
    def add_reflection(self, stage, reflection_text, related_memory_ids=None):
        """Add a reflection on experiences"""
        reflection = {
            'id': len(self.reflections),
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'reflection': reflection_text,
            'related_memories': related_memory_ids or []
        }
        
        self.reflections.append(reflection)
        return reflection['id']
    
    def add_to_narrative(self, stage, narrative_text):
        """Add to the developmental narrative arc"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'stage': stage,
            'narrative': narrative_text
        }
        
        self.narrative_arc.append(entry)
    
    def recall_memories(self, stage=None, event_type=None, min_importance=0):
        """Recall memories matching criteria"""
        filtered = self.memories
        
        if stage:
            filtered = [m for m in filtered if m['stage'] == stage]
        
        if event_type:
            filtered = [m for m in filtered if m['event_type'] == event_type]
        
        if min_importance > 0:
            filtered = [m for m in filtered if m['importance'] >= min_importance]
        
        return sorted(filtered, key=lambda m: m['importance'], reverse=True)
    
    def get_stage_narrative(self, stage):
        """Get the narrative for a specific stage"""
        stage_memories = self.recall_memories(stage=stage)
        stage_reflections = [r for r in self.reflections if r['stage'] == stage]
        stage_narrative = [n for n in self.narrative_arc if n['stage'] == stage]
        
        return {
            'stage': stage,
            'memories': stage_memories,
            'reflections': stage_reflections,
            'narrative': stage_narrative,
            'summary': self.summarize_stage(stage)
        }
    
    def summarize_stage(self, stage):
        """Generate a summary of experiences in a stage"""
        memories = self.recall_memories(stage=stage)
        
        if not memories:
            return f"No memories recorded for {stage} stage"
        
        total_memories = len(memories)
        avg_importance = sum(m['importance'] for m in memories) / total_memories
        
        event_types = {}
        for m in memories:
            event_types[m['event_type']] = event_types.get(m['event_type'], 0) + 1
        
        summary = f"Stage {stage}: {total_memories} memories recorded. "
        summary += f"Average importance: {avg_importance:.1f}/10. "
        
        if event_types:
            most_common = max(event_types.items(), key=lambda x: x[1])
            summary += f"Most common event: {most_common[0]} ({most_common[1]} times). "
        
        return summary
    
    def get_full_narrative(self):
        """Get the complete developmental narrative"""
        return {
            'total_memories': len(self.memories),
            'total_reflections': len(self.reflections),
            'milestones': self.milestones,
            'narrative_arc': self.narrative_arc,
            'stage_summaries': {
                stage: self.summarize_stage(stage)
                for stage in set(m['stage'] for m in self.memories)
            }
        }
    
    def create_learning_story(self):
        """Create a cohesive story of the learning journey"""
        story_parts = []
        
        for entry in self.narrative_arc:
            story_parts.append(f"[{entry['stage']}] {entry['narrative']}")
        
        return "\n\n".join(story_parts)
    
    def save_narrative(self, filepath='knowledge/narrative_memory.json'):
        """Save narrative memory"""
        data = {
            'memories': self.memories,
            'reflections': self.reflections,
            'milestones': self.milestones,
            'narrative_arc': self.narrative_arc,
            'full_narrative': self.get_full_narrative()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
