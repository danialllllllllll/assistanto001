
import numpy as np
import json

class ProgressiveNeuralNetwork:
    """Enhanced neural network with advanced mathematical optimizations"""

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
        
        # IMMUTABLE CORE VALUES LOCK - Cannot be changed by mutations or learning
        self.core_values_lock = {
            'kindness_weight': 1.0,
            'harm_prevention': True,
            'truth_seeking': True,
            'positive_relationships': True,
            'non_harm_threshold': 0.0,
            'locked': True  # Permanent lock flag
        }
        
        # Advanced optimization components
        self.adam_m = []  # First moment (mean)
        self.adam_v = []  # Second moment (variance)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0  # Time step for Adam
        
        # Batch normalization parameters
        self.bn_gamma = []
        self.bn_beta = []
        self.bn_running_mean = []
        self.bn_running_var = []
        
        # Dropout
        self.dropout_rate = 0.0
        self.dropout_masks = []
        
        # Gradient clipping
        self.max_grad_norm = 5.0
        
        # Self-optimization tracking
        self.last_mutation_strategy = []
        
        self.initialize_advanced_params()

    def initialize_weights(self):
        """Initialize weights using He initialization"""
        self.weights = []
        self.biases = []

        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]

        for i in range(len(layer_sizes) - 1):
            # He initialization for better gradient flow
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def initialize_advanced_params(self):
        """Initialize Adam optimizer and batch norm parameters"""
        for i in range(len(self.weights)):
            # Adam optimizer moments
            self.adam_m.append({
                'weights': np.zeros_like(self.weights[i]),
                'biases': np.zeros_like(self.biases[i])
            })
            self.adam_v.append({
                'weights': np.zeros_like(self.weights[i]),
                'biases': np.zeros_like(self.biases[i])
            })
            
            # Batch normalization (except output layer)
            if i < len(self.weights) - 1:
                self.bn_gamma.append(np.ones((1, self.weights[i].shape[1])))
                self.bn_beta.append(np.zeros((1, self.weights[i].shape[1])))
                self.bn_running_mean.append(np.zeros((1, self.weights[i].shape[1])))
                self.bn_running_var.append(np.ones((1, self.weights[i].shape[1])))

    def set_stage_activation(self, active_percent):
        """Scale nodes based on developmental stage"""
        for i, scales in enumerate(self.node_scales):
            scales[:] = active_percent
        
        # Adjust dropout based on stage
        if active_percent < 0.5:
            self.dropout_rate = 0.1  # Light regularization early
        elif active_percent < 0.8:
            self.dropout_rate = 0.2  # More regularization mid-stage
        else:
            self.dropout_rate = 0.3  # Strong regularization late-stage

    def batch_normalize(self, x, layer_idx, training=True):
        """Apply batch normalization"""
        if layer_idx >= len(self.bn_gamma):
            return x
        
        if training:
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            
            # Update running statistics
            momentum = 0.9
            self.bn_running_mean[layer_idx] = momentum * self.bn_running_mean[layer_idx] + (1 - momentum) * mean
            self.bn_running_var[layer_idx] = momentum * self.bn_running_var[layer_idx] + (1 - momentum) * var
        else:
            mean = self.bn_running_mean[layer_idx]
            var = self.bn_running_var[layer_idx]
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        
        # Scale and shift
        return self.bn_gamma[layer_idx] * x_norm + self.bn_beta[layer_idx]

    def apply_dropout(self, x, training=True):
        """Apply dropout for regularization"""
        if not training or self.dropout_rate == 0:
            return x, np.ones_like(x)
        
        mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape) / (1 - self.dropout_rate)
        return x * mask, mask

    def leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU to prevent dead neurons"""
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        """Derivative of Leaky ReLU"""
        return np.where(x > 0, 1, alpha)

    def swish(self, x):
        """Swish activation function (x * sigmoid(x))"""
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))

    def swish_derivative(self, x):
        """Derivative of Swish"""
        sig = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return sig + x * sig * (1 - sig)

    def softmax(self, x):
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X, training=True):
        """Forward pass with batch norm and dropout"""
        self.activations = [X]
        self.z_values = []
        self.dropout_masks = []
        
        activation = X

        for i in range(len(self.weights) - 1):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Batch normalization
            z = self.batch_normalize(z, i, training)
            
            # Activation function (Swish for better gradients)
            activation = self.swish(z)
            
            # Apply node scaling
            if i < len(self.node_scales):
                activation = activation * self.node_scales[i]
            
            # Dropout
            activation, mask = self.apply_dropout(activation, training)
            self.dropout_masks.append(mask)
            
            self.activations.append(activation)

        # Output layer (no dropout, no batch norm)
        z_out = np.dot(activation, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_out)
        output = self.softmax(z_out)
        self.activations.append(output)

        return output

    def clip_gradients(self, gradients):
        """Clip gradients to prevent exploding gradients"""
        total_norm = 0
        for grad in gradients:
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            return [grad * clip_coef for grad in gradients]
        return gradients

    def backward(self, X, y, learning_rate):
        """Backpropagation with Adam optimizer"""
        m = X.shape[0]
        self.t += 1

        # One-hot encode targets
        y_one_hot = np.zeros((m, self.output_size))
        y_one_hot[np.arange(m), y.astype(int)] = 1

        # Output layer gradient
        delta = self.activations[-1] - y_one_hot

        weight_gradients = []
        bias_gradients = []

        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            weight_grad = np.dot(self.activations[i].T, delta) / m
            bias_grad = np.sum(delta, axis=0, keepdims=True) / m

            weight_gradients.insert(0, weight_grad)
            bias_gradients.insert(0, bias_grad)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                
                # Reverse dropout
                if i - 1 < len(self.dropout_masks):
                    delta = delta * self.dropout_masks[i - 1]
                
                # Reverse node scaling
                if i - 1 < len(self.node_scales):
                    delta = delta * self.node_scales[i - 1]
                
                # Reverse activation
                delta = delta * self.swish_derivative(self.z_values[i - 1])

        # Clip gradients
        weight_gradients = self.clip_gradients(weight_gradients)
        bias_gradients = self.clip_gradients(bias_gradients)

        # Adam optimizer update
        for i in range(len(self.weights)):
            # Update biased first moment estimate
            self.adam_m[i]['weights'] = self.beta1 * self.adam_m[i]['weights'] + (1 - self.beta1) * weight_gradients[i]
            self.adam_m[i]['biases'] = self.beta1 * self.adam_m[i]['biases'] + (1 - self.beta1) * bias_gradients[i]
            
            # Update biased second moment estimate
            self.adam_v[i]['weights'] = self.beta2 * self.adam_v[i]['weights'] + (1 - self.beta2) * (weight_gradients[i] ** 2)
            self.adam_v[i]['biases'] = self.beta2 * self.adam_v[i]['biases'] + (1 - self.beta2) * (bias_gradients[i] ** 2)
            
            # Bias correction
            m_w_corrected = self.adam_m[i]['weights'] / (1 - self.beta1 ** self.t)
            m_b_corrected = self.adam_m[i]['biases'] / (1 - self.beta1 ** self.t)
            v_w_corrected = self.adam_v[i]['weights'] / (1 - self.beta2 ** self.t)
            v_b_corrected = self.adam_v[i]['biases'] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            self.weights[i] -= learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)
            self.biases[i] -= learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)

    def predict(self, X):
        """Make predictions (inference mode)"""
        output = self.forward(X, training=False)
        return np.argmax(output, axis=1)

    def get_confidence(self, X):
        """Get prediction confidence"""
        output = self.forward(X, training=False)
        return np.max(output, axis=1)

    def save_weights(self, filepath):
        """Save network weights and optimizer state"""
        data = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'node_scales': [s.tolist() for s in self.node_scales],
            'adam_m': [{'weights': m['weights'].tolist(), 'biases': m['biases'].tolist()} for m in self.adam_m],
            'adam_v': [{'weights': v['weights'].tolist(), 'biases': v['biases'].tolist()} for v in self.adam_v],
            'bn_gamma': [g.tolist() for g in self.bn_gamma],
            'bn_beta': [b.tolist() for b in self.bn_beta],
            't': self.t
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load_weights(self, filepath):
        """Load network weights and optimizer state"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.weights = [np.array(w) for w in data['weights']]
        self.biases = [np.array(b) for b in data['biases']]
        self.node_scales = [np.array(s) for s in data['node_scales']]
        
        if 'adam_m' in data:
            self.adam_m = [{'weights': np.array(m['weights']), 'biases': np.array(m['biases'])} for m in data['adam_m']]
            self.adam_v = [{'weights': np.array(v['weights']), 'biases': np.array(v['biases'])} for v in data['adam_v']]
            self.t = data.get('t', 0)
