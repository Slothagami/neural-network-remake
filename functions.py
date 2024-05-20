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
    
class MSE:
    def forward(self, target, pred):
        return np.mean(np.square(target - pred))
    
    def backprop(self, target, pred):
        return -2 * (target - pred) / target.size
