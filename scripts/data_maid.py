
"""
Data Maid - Intelligent cleanup of redundant training data
Optimizes storage while preserving important learning milestones
"""

import os
import json
import zipfile
import shutil
from datetime import datetime
from pathlib import Path

class DataMaid:
    def __init__(self, archives_dir='training_archives'):
        self.archives_dir = archives_dir
        self.cleanup_report = {
            'start_time': datetime.now().isoformat(),
            'files_deleted': 0,
            'space_freed': 0,
            'redundancies_found': 0,
            'kept_milestones': []
        }
    
    def analyze_redundancy(self, data1, data2, threshold=0.95):
        """Check if two data points are redundant (>95% similarity)"""
        if not isinstance(data1, dict) or not isinstance(data2, dict):
            return False
        
        # Compare understanding scores
        u1 = data1.get('understanding', 0)
        u2 = data2.get('understanding', 0)
        
        # If understanding is nearly identical, check other metrics
        if abs(u1 - u2) < 0.01:
            a1 = data1.get('accuracy', 0)
            a2 = data2.get('accuracy', 0)
            c1 = data1.get('confidence', 0)
            c2 = data2.get('confidence', 0)
            
            similarity = 1 - (abs(a1-a2) + abs(c1-c2)) / 2
            return similarity > threshold
        
        return False
    
    def is_milestone(self, data, phase_data):
        """Determine if data point is a significant milestone"""
        understanding = data.get('understanding', 0)
        
        # First iteration of each phase
        if data.get('iteration', 0) == 0:
            return True
        
        # High understanding achievements
        if understanding >= 0.999:
            return True
        
        # Significant improvement from previous
        iteration = data.get('iteration', 0)
        if iteration > 0:
            prev_u = phase_data[iteration - 1].get('understanding', 0) if iteration - 1 < len(phase_data) else 0
            if understanding - prev_u > 0.05:  # 5% jump
                return True
        
        # Every 100th iteration as checkpoint
        if iteration % 100 == 0:
            return True
        
        return False
    
    def clean_phase_archives(self, phase_name):
        """Clean archives for a specific phase"""
        phase_path = os.path.join(self.archives_dir, phase_name)
        
        if not os.path.exists(phase_path):
            return
        
        batch_files = sorted([f for f in os.listdir(phase_path) if f.startswith('batch_') and f.endswith('.zip')])
        
        for batch_file in batch_files:
            batch_path = os.path.join(phase_path, batch_file)
            temp_extract = os.path.join(phase_path, f'temp_extract_{batch_file}')
            
            try:
                # Extract batch
                with zipfile.ZipFile(batch_path, 'r') as zipf:
                    zipf.extractall(temp_extract)
                
                # Load all generation data
                generation_files = sorted([f for f in os.listdir(temp_extract) if f.startswith('generation_')])
                phase_data = []
                
                for gen_file in generation_files:
                    gen_path = os.path.join(temp_extract, gen_file)
                    with open(gen_path, 'r') as f:
                        phase_data.append(json.load(f))
                
                # Identify redundancies
                keep_indices = set()
                for i, data in enumerate(phase_data):
                    # Always keep if it's a milestone
                    if self.is_milestone(data, phase_data):
                        keep_indices.add(i)
                        continue
                    
                    # Check for redundancy with previous
                    if i > 0 and i - 1 in keep_indices:
                        if not self.analyze_redundancy(phase_data[i-1], data):
                            keep_indices.add(i)
                    elif i == 0:
                        keep_indices.add(i)
                
                # Count redundancies
                redundant_count = len(phase_data) - len(keep_indices)
                self.cleanup_report['redundancies_found'] += redundant_count
                
                # If we can save space, repack
                if redundant_count > 0:
                    new_batch_path = batch_path + '.cleaned'
                    
                    with zipfile.ZipFile(new_batch_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for i in sorted(keep_indices):
                            gen_file = generation_files[i]
                            gen_path = os.path.join(temp_extract, gen_file)
                            zipf.write(gen_path, gen_file)
                    
                    # Get size savings
                    original_size = os.path.getsize(batch_path)
                    new_size = os.path.getsize(new_batch_path)
                    space_saved = original_size - new_size
                    
                    # Replace if significant savings
                    if space_saved > 1024:  # At least 1KB saved
                        os.remove(batch_path)
                        os.rename(new_batch_path, batch_path)
                        self.cleanup_report['space_freed'] += space_saved
                        self.cleanup_report['files_deleted'] += redundant_count
                    else:
                        os.remove(new_batch_path)
                
                # Cleanup temp
                shutil.rmtree(temp_extract)
                
            except Exception as e:
                print(f"Error cleaning {batch_file}: {e}")
                if os.path.exists(temp_extract):
                    shutil.rmtree(temp_extract)
    
    def clean_all_phases(self):
        """Clean all phase archives"""
        phases = ['baby_steps', 'toddler', 'pre-k', 'elementary', 'teen', 'scholar', 'thinker']
        
        for phase in phases:
            print(f"Cleaning {phase}...")
            self.clean_phase_archives(phase)
        
        self.cleanup_report['end_time'] = datetime.now().isoformat()
        
        # Save report
        report_path = os.path.join(self.archives_dir, 'cleanup_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.cleanup_report, f, indent=2)
        
        print(f"\nCleanup Report:")
        print(f"  Redundancies removed: {self.cleanup_report['redundancies_found']}")
        print(f"  Files deleted: {self.cleanup_report['files_deleted']}")
        print(f"  Space freed: {self.cleanup_report['space_freed'] / 1024 / 1024:.2f} MB")
        print(f"  Report saved to: {report_path}")

if __name__ == "__main__":
    maid = DataMaid()
    maid.clean_all_phases()
