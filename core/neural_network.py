import numpy as np
import json

class ProgressiveNeuralNetwork:
    """Neural network with progressive node activation through developmental stages"""
    
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        self.initialize_weights()
        self.node_scales = []
        for hidden_size in hidden_sizes:
            self.node_scales.append(np.ones(hidden_size))
        
        self.activations = []
        self.z_values = []
        
    def initialize_weights(self):
        """Initialize weights using He initialization for better gradient flow"""
        self.weights = []
        self.biases = []
        
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def set_stage_activation(self, active_percent):
        """Scale nodes based on developmental stage (not masking, scaling)"""
        for i, scales in enumerate(self.node_scales):
            num_nodes = len(scales)
            num_active = int(num_nodes * active_percent)
            
            scales[:] = 0.0
            if num_active > 0:
                active_indices = np.random.choice(num_nodes, num_active, replace=False)
                scales[active_indices] = 1.0
    
    def sigmoid(self, x):
        """Sigmoid activation with numerical stability"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def softmax(self, x):
        """Softmax for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward pass with scaled node activation"""
        self.activations = [X]
        self.z_values = []
        
        activation = X
        
        for i in range(len(self.weights) - 1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = self.sigmoid(z)
            
            if i < len(self.node_scales):
                activation = activation * self.node_scales[i]
            
            self.activations.append(activation)
        
        z_out = np.dot(activation, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_out)
        output = self.softmax(z_out)
        self.activations.append(output)
        
        return output
    
    def backward(self, X, y, learning_rate):
        """Backpropagation with gradient descent"""
        m = X.shape[0]
        
        y_one_hot = np.zeros((m, self.output_size))
        y_one_hot[np.arange(m), y.astype(int)] = 1
        
        delta = self.activations[-1] - y_one_hot
        
        weight_gradients = []
        bias_gradients = []
        
        for i in range(len(self.weights) - 1, -1, -1):
            weight_grad = np.dot(self.activations[i].T, delta) / m
            bias_grad = np.sum(delta, axis=0, keepdims=True) / m
            
            weight_gradients.insert(0, weight_grad)
            bias_gradients.insert(0, bias_grad)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                
                if i - 1 < len(self.node_scales):
                    delta = delta * self.node_scales[i - 1]
                
                delta = delta * self.sigmoid_derivative(self.z_values[i - 1])
        
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * weight_gradients[i]
            self.biases[i] -= learning_rate * bias_gradients[i]
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def get_confidence(self, X):
        """Get prediction confidence"""
        output = self.forward(X)
        return np.max(output, axis=1)
    
    def save_weights(self, filepath):
        """Save network weights"""
        data = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'node_scales': [s.tolist() for s in self.node_scales]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load_weights(self, filepath):
        """Load network weights"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.weights = [np.array(w) for w in data['weights']]
        self.biases = [np.array(b) for b in data['biases']]
        self.node_scales = [np.array(s) for s in data['node_scales']]
