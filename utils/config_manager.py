
"""Configuration management for the AI system"""

import json
import os
from typing import Dict, Any

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_dir: str = 'configs'):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
        self._configs = {}
    
    def load_all_configs(self):
        """Load all configuration files"""
        config_files = {
            'stages': 'stage_config.json',
            'core_values': 'core_values.json'
        }
        
        for key, filename in config_files.items():
            filepath = os.path.join(self.config_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self._configs[key] = json.load(f)
        
        return self._configs
    
    def get(self, key: str, default: Any = None):
        """Get configuration by key"""
        return self._configs.get(key, default)
    
    def save_config(self, key: str, data: Dict, filename: str = None):
        """Save configuration to file"""
        if filename is None:
            filename = f"{key}_config.json"
        
        filepath = os.path.join(self.config_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self._configs[key] = data
