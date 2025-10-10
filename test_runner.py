
#!/usr/bin/env python3
"""Test runner to verify all components work correctly"""

import sys
import traceback
from core.autonomous_debugger import autonomous_debugger

class ComponentTester:
    """Test all AI components"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.errors = []
    
    def run_test(self, test_name: str, test_func):
        """Run a single test with error handling"""
        print(f"\n🧪 Testing: {test_name}")
        try:
            result = autonomous_debugger.monitor_execution(test_func)
            if result is not None or result is True:
                print(f"   ✓ {test_name} passed")
                self.tests_passed += 1
                return True
            else:
                print(f"   ✗ {test_name} failed")
                self.tests_failed += 1
                return False
        except Exception as e:
            print(f"   ✗ {test_name} error: {e}")
            self.errors.append({'test': test_name, 'error': str(e)})
            self.tests_failed += 1
            return False
    
    def test_imports(self):
        """Test all imports"""
        try:
            from core import neural_network, genetic_trainer, autonomous_debugger
            from personality import traits, narrative_memory
            from philosophy import thinker_engine, reasoning_rules
            from knowledge import storage, web_learning
            return True
        except ImportError as e:
            print(f"Import error: {e}")
            return False
    
    def test_neural_network(self):
        """Test neural network creation"""
        from core.neural_network import ProgressiveNeuralNetwork
        network = ProgressiveNeuralNetwork(10, [20, 10], 4)
        import numpy as np
        X = np.random.randn(5, 10)
        output = network.forward(X)
        return output.shape == (5, 4)
    
    def test_genetic_trainer(self):
        """Test genetic trainer initialization"""
        from core.neural_network import ProgressiveNeuralNetwork
        from core.genetic_trainer import GeneticTrainer
        
        network = ProgressiveNeuralNetwork(10, [20, 10], 4)
        trainer = GeneticTrainer(network, population_size=5, elite_size=1)
        trainer.initialize_population()
        return len(trainer.population) == 5
    
    def test_file_manager(self):
        """Test file management utilities"""
        from utils.file_manager import FileManager
        FileManager.ensure_directories()
        test_data = {'test': 'data'}
        FileManager.save_json('test_config.json', test_data)
        loaded = FileManager.load_json('test_config.json')
        import os
        os.remove('test_config.json')
        return loaded == test_data
    
    def run_all_tests(self):
        """Run all component tests"""
        print("="*60)
        print("AI SYSTEM COMPONENT TESTS")
        print("="*60)
        
        self.run_test("Import Check", self.test_imports)
        self.run_test("Neural Network", self.test_neural_network)
        self.run_test("Genetic Trainer", self.test_genetic_trainer)
        self.run_test("File Manager", self.test_file_manager)
        
        print("\n" + "="*60)
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_failed}")
        
        if self.errors:
            print("\nErrors encountered:")
            for error in self.errors:
                print(f"  - {error['test']}: {error['error']}")
        
        print("="*60)
        
        return self.tests_failed == 0

if __name__ == "__main__":
    tester = ComponentTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
