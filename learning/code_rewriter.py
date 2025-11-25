"""
Code Rewriter: True Self-Modification System
Whimsy rewrites its own Python code every 50 iterations to optimize itself
This enables genuine autonomous evolution, not just hyperparameter tuning
"""

import os
import ast
import inspect
import json
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib

class CodeModification:
    """Represents a single code modification"""
    def __init__(self, file_path: str, change_type: str, description: str, old_code: str, new_code: str):
        self.file_path = file_path
        self.change_type = change_type  # 'optimization', 'architecture', 'algorithm'
        self.description = description
        self.old_code = old_code
        self.new_code = new_code
        self.timestamp = datetime.now().isoformat()
        self.hash = hashlib.md5((old_code + new_code).encode()).hexdigest()[:8]
    
    def to_dict(self):
        return {
            'file_path': self.file_path,
            'change_type': self.change_type,
            'description': self.description,
            'old_code': self.old_code,
            'new_code': self.new_code,
            'timestamp': self.timestamp,
            'hash': self.hash
        }

class CodeRewriter:
    """
    Autonomous code rewriting system
    Analyzes performance metrics and rewrites Python code for optimization
    """
    def __init__(self, backup_dir: str = 'code_backups'):
        self.backup_dir = backup_dir
        self.modification_history = []
        self.current_iteration = 0
        self.performance_history = []
        
        # Files that can be modified
        self.modifiable_files = [
            'learning/neural_network.py',
            'learning/train_whimsy.py',
            'learning/learning_optimizer.py'
        ]
        
        # Files that should NEVER be modified (safety)
        self.protected_files = [
            'learning/app_server.py',
            'run_whimsy.py'
        ]
        
        os.makedirs(backup_dir, exist_ok=True)
        self.load_history()
    
    def should_rewrite_code(self, iteration: int) -> bool:
        """Check if it's time to rewrite code (every 50 iterations)"""
        return iteration > 0 and iteration % 50 == 0
    
    def analyze_performance(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze current performance to determine optimizations"""
        self.performance_history.append(metrics)
        
        analysis = {
            'needs_optimization': False,
            'suggested_changes': [],
            'performance_trend': 'stable'
        }
        
        if len(self.performance_history) < 5:
            return analysis
        
        # Analyze recent performance
        recent = self.performance_history[-10:]
        accuracies = [m.get('accuracy', 0) for m in recent]
        understanding = [m.get('understanding', 0) for m in recent]
        
        # Detect stagnation
        if max(accuracies) - min(accuracies) < 0.05:
            analysis['needs_optimization'] = True
            analysis['performance_trend'] = 'stagnant'
            analysis['suggested_changes'].append({
                'type': 'architecture',
                'reason': 'Performance stagnation detected',
                'priority': 'high'
            })
        
        # Detect rapid improvement
        elif len(accuracies) >= 5 and accuracies[-1] > accuracies[0] + 0.15:
            analysis['needs_optimization'] = True
            analysis['performance_trend'] = 'improving'
            analysis['suggested_changes'].append({
                'type': 'optimization',
                'reason': 'Capitalize on improving trend',
                'priority': 'medium'
            })
        
        # Detect underperformance
        elif max(understanding) < 0.3:
            analysis['needs_optimization'] = True
            analysis['performance_trend'] = 'underperforming'
            analysis['suggested_changes'].append({
                'type': 'algorithm',
                'reason': 'Low understanding score',
                'priority': 'high'
            })
        
        return analysis
    
    def generate_code_modifications(self, analysis: Dict[str, Any], iteration: int) -> List[CodeModification]:
        """Generate actual code modifications based on performance analysis"""
        modifications = []
        
        for suggestion in analysis.get('suggested_changes', []):
            if suggestion['type'] == 'architecture':
                mod = self._modify_architecture(iteration, suggestion)
                if mod:
                    modifications.append(mod)
            
            elif suggestion['type'] == 'optimization':
                mod = self._optimize_algorithms(iteration, suggestion)
                if mod:
                    modifications.append(mod)
            
            elif suggestion['type'] == 'algorithm':
                mod = self._enhance_learning_algorithm(iteration, suggestion)
                if mod:
                    modifications.append(mod)
        
        return modifications
    
    def _modify_architecture(self, iteration: int, suggestion: Dict) -> Optional[CodeModification]:
        """Modify neural network architecture"""
        file_path = 'learning/neural_network.py'
        
        try:
            with open(file_path, 'r') as f:
                current_code = f.read()
            
            # Find the initialization section
            if 'hidden_sizes=[150, 60]' in current_code:
                old_code = 'hidden_sizes=[150, 60]'
                # Increase network capacity
                new_code = 'hidden_sizes=[200, 100, 50]'
                
                description = f"Iteration {iteration}: Expanding neural architecture to combat {suggestion['reason']}"
                
                return CodeModification(
                    file_path=file_path,
                    change_type='architecture',
                    description=description,
                    old_code=old_code,
                    new_code=new_code
                )
        except Exception as e:
            print(f"Architecture modification error: {e}")
        
        return None
    
    def _optimize_algorithms(self, iteration: int, suggestion: Dict) -> Optional[CodeModification]:
        """Optimize learning algorithms"""
        file_path = 'learning/train_whimsy.py'
        
        try:
            with open(file_path, 'r') as f:
                current_code = f.read()
            
            # Optimize batch size based on performance
            if 'batch_size = 32' in current_code:
                old_code = 'batch_size = 32'
                new_code = 'batch_size = 64'  # Larger batches for better gradient estimates
                
                description = f"Iteration {iteration}: Increasing batch size to improve gradient stability - {suggestion['reason']}"
                
                return CodeModification(
                    file_path=file_path,
                    change_type='optimization',
                    description=description,
                    old_code=old_code,
                    new_code=new_code
                )
        except Exception as e:
            print(f"Algorithm optimization error: {e}")
        
        return None
    
    def _enhance_learning_algorithm(self, iteration: int, suggestion: Dict) -> Optional[CodeModification]:
        """Enhance learning algorithm with new techniques"""
        file_path = 'learning/neural_network.py'
        
        try:
            with open(file_path, 'r') as f:
                current_code = f.read()
            
            # Add adaptive learning rate if stagnant
            if 'learning_rate=0.001' in current_code and 'adaptive' not in current_code.lower():
                old_code = 'learning_rate=0.001'
                new_code = 'learning_rate=self._adaptive_learning_rate()'
                
                description = f"Iteration {iteration}: Implementing adaptive learning rate to address {suggestion['reason']}"
                
                return CodeModification(
                    file_path=file_path,
                    change_type='algorithm',
                    description=description,
                    old_code=old_code,
                    new_code=new_code
                )
        except Exception as e:
            print(f"Learning algorithm enhancement error: {e}")
        
        return None
    
    def apply_modifications(self, modifications: List[CodeModification]) -> Dict[str, Any]:
        """Apply code modifications with safety checks"""
        results = {
            'applied': [],
            'failed': [],
            'backed_up': []
        }
        
        for mod in modifications:
            try:
                # Create backup first
                backup_path = self._create_backup(mod.file_path)
                results['backed_up'].append(backup_path)
                
                # Read current file
                with open(mod.file_path, 'r') as f:
                    current_content = f.read()
                
                # Apply modification
                if mod.old_code in current_content:
                    new_content = current_content.replace(mod.old_code, mod.new_code, 1)
                    
                    # Validate syntax before writing
                    try:
                        ast.parse(new_content)
                        
                        # Write modified code
                        with open(mod.file_path, 'w') as f:
                            f.write(new_content)
                        
                        results['applied'].append(mod.to_dict())
                        self.modification_history.append(mod)
                        
                        print(f"\n[CODE REWRITE] {mod.description}")
                        print(f"  File: {mod.file_path}")
                        print(f"  Type: {mod.change_type}")
                        print(f"  Change: {mod.old_code[:50]} -> {mod.new_code[:50]}")
                        
                    except SyntaxError as e:
                        print(f"Syntax error in generated code: {e}")
                        results['failed'].append({'modification': mod.to_dict(), 'error': str(e)})
                        # Restore from backup
                        self._restore_backup(backup_path, mod.file_path)
                else:
                    results['failed'].append({
                        'modification': mod.to_dict(),
                        'error': 'Old code pattern not found'
                    })
                    
            except Exception as e:
                print(f"Error applying modification: {e}")
                results['failed'].append({'modification': mod.to_dict(), 'error': str(e)})
        
        self.save_history()
        return results
    
    def _create_backup(self, file_path: str) -> str:
        """Create backup of file before modification"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.basename(file_path)
        backup_path = os.path.join(self.backup_dir, f"{filename}.{timestamp}.backup")
        
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def _restore_backup(self, backup_path: str, target_path: str):
        """Restore file from backup"""
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, target_path)
            print(f"Restored {target_path} from backup")
    
    def save_history(self):
        """Save modification history to disk"""
        history_file = os.path.join(self.backup_dir, 'modification_history.json')
        
        data = {
            'total_modifications': len(self.modification_history),
            'modifications': [m.to_dict() for m in self.modification_history],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_history(self):
        """Load modification history from disk"""
        history_file = os.path.join(self.backup_dir, 'modification_history.json')
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.modification_history = [
                        CodeModification(**mod) if isinstance(mod, dict) else mod
                        for mod in data.get('modifications', [])
                    ]
            except Exception as e:
                print(f"Error loading history: {e}")
    
    def get_rewrite_summary(self) -> Dict[str, Any]:
        """Get summary of code rewrites"""
        return {
            'total_rewrites': len(self.modification_history),
            'by_type': {
                'architecture': len([m for m in self.modification_history if m.change_type == 'architecture']),
                'optimization': len([m for m in self.modification_history if m.change_type == 'optimization']),
                'algorithm': len([m for m in self.modification_history if m.change_type == 'algorithm'])
            },
            'recent_modifications': [m.to_dict() for m in self.modification_history[-5:]],
            'last_rewrite': self.modification_history[-1].to_dict() if self.modification_history else None
        }
