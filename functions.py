import numpy as np

class Sigmoid:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return 1 / (1 + np.exp(-x))
    
    def backprop(self, d_err, lr):
        return d_err * self.forward(self.input) * (1 - self.forward(self.input))
    
class Tanh:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.tanh(x)
    
    def backprop(self, d_err, lr):
        return d_err * ( 1 - np.square(np.tanh(self.input)) )
    
class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backprop(self, d_err, lr):
        return d_err * (self.input > 0).astype(np.float32)
    
class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, x):
        x  -= np.max(x) # shift input to avoid overflows
        exp = np.exp(x)
        self.output = exp / np.sum(exp)
        return self.output
    
    def backprop(self, d_err, lr):
        # can be optimized with a different formula/methods.
        stack_size = self.output.size
        y_mat = np.tile(self.output, stack_size)

        derivative = y_mat * (np.identity(stack_size) - y_mat.T)

        return np.dot(derivative, d_err)
    
class MSE:
    def forward(self, target, pred):
        return np.mean(np.square(target - pred))
    
    def backprop(self, target, pred):
        return -2 * (target - pred) / target.size

EPS = 1e-16
class CCE: # Categorical Cross Entropy
    def forward(self, target, pred):
        return -np.sum(target * np.log(pred + EPS)) / target.size
    
    def backprop(self, target, pred):
        return (pred - target) / target.size
