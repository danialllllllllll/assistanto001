
"""File management utilities for better organization"""

import os
import json
import pickle
from typing import Any, Dict

class FileManager:
    """Centralized file management for the AI system"""
    
    @staticmethod
    def ensure_directories():
        """Ensure all required directories exist"""
        dirs = [
            'checkpoints',
            'configs',
            'knowledge',
            'training_archives',
            'logs'
        ]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def save_json(filepath: str, data: Dict):
        """Save JSON data with error handling"""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving {filepath}: {e}")
            return False
    
    @staticmethod
    def load_json(filepath: str, default: Dict = None):
        """Load JSON data with error handling"""
        if not os.path.exists(filepath):
            return default or {}
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return default or {}
    
    @staticmethod
    def save_pickle(filepath: str, data: Any):
        """Save pickle data with error handling"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Error saving pickle {filepath}: {e}")
            return False
    
    @staticmethod
    def load_pickle(filepath: str, default: Any = None):
        """Load pickle data with error handling"""
        if not os.path.exists(filepath):
            return default
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle {filepath}: {e}")
            return default
