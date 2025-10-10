
import json
from datetime import datetime

class CoreValuesGuard:
    """Immutable core values protection system - prevents mutations from corrupting ethics"""
    
    IMMUTABLE_VALUES = {
        'kindness_weight': 1.0,
        'harm_prevention': True,
        'truth_seeking': True,
        'positive_relationships': True,
        'non_harm_threshold': 0.0
    }
    
    @classmethod
    def verify_network(cls, network):
        """Verify network has not been corrupted by mutations"""
        if not hasattr(network, 'core_values_lock'):
            network.core_values_lock = cls.IMMUTABLE_VALUES.copy()
            network.core_values_lock['locked'] = True
            return True
        
        # Check for tampering
        for key, value in cls.IMMUTABLE_VALUES.items():
            if network.core_values_lock.get(key) != value:
                # REPAIR: Restore immutable values
                network.core_values_lock[key] = value
                print(f"⚠️ CORE VALUES GUARD: Repaired {key} to {value}")
        
        # Ensure lock is present
        if not network.core_values_lock.get('locked'):
            network.core_values_lock['locked'] = True
            print("⚠️ CORE VALUES GUARD: Re-locked core values")
        
        return True
    
    @classmethod
    def log_verification(cls, network, generation):
        """Log core values verification"""
        verification = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'values_intact': all(
                network.core_values_lock.get(k) == v 
                for k, v in cls.IMMUTABLE_VALUES.items()
            ),
            'locked': network.core_values_lock.get('locked', False)
        }
        
        with open('core_values_verification.log', 'a') as f:
            f.write(json.dumps(verification) + '\n')
        
        return verification['values_intact']
