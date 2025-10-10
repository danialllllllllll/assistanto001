
#!/usr/bin/env python3
"""Verify all dependencies and system requirements are met"""

import sys

def check_imports():
    """Check if all required modules can be imported"""
    required_modules = [
        'numpy',
        'flask',
        'flask_cors',
        'pickle',
        'json',
        'datetime',
        'threading',
        'time',
        'os'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\n❌ Missing modules: {', '.join(missing)}")
        print("Run: upm add " + " ".join(missing))
        return False
    
    print("\n✓ All core dependencies available")
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'dataset_features.pkl',
        'outputs.pkl',
        'configs/stage_config.json',
        'configs/core_values.json'
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            missing.append(file)
    
    if missing:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        return False
    
    print("\n✓ All required files present")
    return True

if __name__ == '__main__':
    import os
    print("="*60)
    print("VERIFYING SYSTEM SETUP")
    print("="*60)
    
    print("\nChecking Python modules...")
    modules_ok = check_imports()
    
    print("\nChecking required files...")
    files_ok = check_files()
    
    print("\n" + "="*60)
    if modules_ok and files_ok:
        print("✓ SYSTEM READY - All checks passed!")
        print("Run: python train_advanced_ai.py")
        sys.exit(0)
    else:
        print("❌ SYSTEM NOT READY - Please fix issues above")
        sys.exit(1)
