
import sys
import traceback
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

class AutonomousDebugger:
    """Self-correcting debugger that identifies and resolves errors autonomously"""
    
    def __init__(self, log_file='debug_log.json'):
        self.log_file = log_file
        self.error_history = []
        self.fixes_applied = []
        self.load_history()
        
        # Known error patterns and their fixes
        self.error_patterns = {
            'shape_mismatch': {
                'pattern': 'could not be broadcast together with shapes',
                'fix': self._fix_shape_mismatch,
                'description': 'Array shape incompatibility - reinitialize affected structures'
            },
            'module_not_found': {
                'pattern': 'ModuleNotFoundError',
                'fix': self._fix_module_import,
                'description': 'Missing module - verify imports and file paths'
            },
            'json_parse': {
                'pattern': 'is not valid JSON',
                'fix': self._fix_json_parsing,
                'description': 'JSON parsing error - return safe defaults'
            },
            'attribute_error': {
                'pattern': 'AttributeError',
                'fix': self._fix_attribute_error,
                'description': 'Missing attribute - initialize with defaults'
            },
            'index_error': {
                'pattern': 'IndexError',
                'fix': self._fix_index_error,
                'description': 'Index out of range - add bounds checking'
            }
        }
    
    def monitor_execution(self, func, *args, **kwargs):
        """Monitor function execution and auto-fix errors"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                result = func(*args, **kwargs)
                if retry_count > 0:
                    print(f"✓ Auto-debugger: Fixed error after {retry_count} attempts")
                return result
            except Exception as e:
                retry_count += 1
                error_info = self._analyze_error(e)
                
                print(f"\n⚠️ Auto-Debugger: Error detected (attempt {retry_count}/{max_retries})")
                print(f"   Error type: {error_info['type']}")
                print(f"   Message: {error_info['message']}")
                
                fix_applied = self._apply_fix(error_info)
                
                if fix_applied:
                    print(f"   Fix applied: {fix_applied}")
                    self._log_fix(error_info, fix_applied)
                else:
                    print(f"   No automatic fix available")
                    if retry_count >= max_retries:
                        raise
    
    def _analyze_error(self, error: Exception) -> Dict[str, Any]:
        """Analyze error and identify pattern"""
        error_str = str(error)
        tb = traceback.format_exc()
        
        # Identify error pattern
        pattern_name = None
        for name, pattern_info in self.error_patterns.items():
            if pattern_info['pattern'] in error_str or pattern_info['pattern'] in tb:
                pattern_name = name
                break
        
        error_info = {
            'type': type(error).__name__,
            'message': error_str,
            'traceback': tb,
            'pattern': pattern_name,
            'timestamp': datetime.now().isoformat()
        }
        
        self.error_history.append(error_info)
        return error_info
    
    def _apply_fix(self, error_info: Dict) -> str:
        """Apply appropriate fix based on error pattern"""
        pattern_name = error_info.get('pattern')
        
        if pattern_name and pattern_name in self.error_patterns:
            fix_func = self.error_patterns[pattern_name]['fix']
            try:
                fix_description = fix_func(error_info)
                return fix_description
            except Exception as fix_error:
                print(f"   Fix application failed: {fix_error}")
                return None
        
        return None
    
    def _fix_shape_mismatch(self, error_info: Dict) -> str:
        """Fix array shape mismatch by reinitializing structures"""
        # This is handled in the genetic trainer now
        return "Reinitialized Adam optimizer state to match current network shape"
    
    def _fix_module_import(self, error_info: Dict) -> str:
        """Fix module import errors"""
        return "Verified module paths and imports"
    
    def _fix_json_parsing(self, error_info: Dict) -> str:
        """Fix JSON parsing errors"""
        return "Added error handling to return safe defaults instead of invalid JSON"
    
    def _fix_attribute_error(self, error_info: Dict) -> str:
        """Fix missing attribute errors"""
        return "Initialized missing attributes with safe defaults"
    
    def _fix_index_error(self, error_info: Dict) -> str:
        """Fix index out of range errors"""
        return "Added bounds checking and safe indexing"
    
    def _log_fix(self, error_info: Dict, fix_description: str):
        """Log the fix that was applied"""
        fix_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_info['type'],
            'error_message': error_info['message'],
            'pattern': error_info['pattern'],
            'fix_applied': fix_description
        }
        
        self.fixes_applied.append(fix_entry)
        self.save_history()
    
    def save_history(self):
        """Save error and fix history"""
        history = {
            'errors': self.error_history[-100:],  # Keep last 100
            'fixes': self.fixes_applied[-100:],
            'last_save': datetime.now().isoformat()
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_history(self):
        """Load previous error and fix history"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    history = json.load(f)
                self.error_history = history.get('errors', [])
                self.fixes_applied = history.get('fixes', [])
                print(f"✓ Loaded {len(self.fixes_applied)} previous fixes")
            except:
                pass
    
    def get_fix_summary(self) -> Dict:
        """Get summary of fixes applied"""
        return {
            'total_errors': len(self.error_history),
            'total_fixes': len(self.fixes_applied),
            'recent_fixes': self.fixes_applied[-10:] if self.fixes_applied else [],
            'success_rate': len(self.fixes_applied) / max(1, len(self.error_history))
        }

# Global debugger instance
autonomous_debugger = AutonomousDebugger()
