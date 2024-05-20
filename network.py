import numpy as np

class Network:
    def __init__(self, error_func, lr):
        self.layers = []
        self.error_func = error_func
        self.output = None
        self.lr = lr

    def config(self, layers, type, activation):
        for in_size, out_size in zip(layers, layers[1:]):
            self.add(type(in_size, out_size))
            self.add(activation())

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)

        self.output = output
        return output
    
    def train_sample(self, input, target):
        self.forward(input) 
        d_error = self.output_error(target)

        for layer in reversed(self.layers):
            d_error = layer.backprop(d_error, self.lr)

        return self.display_error(target)
    
    def output_error(self, target):
        return self.error_func.backprop(target, self.output)
    
    def display_error(self, target):
        return self.error_func.forward(target, self.output)


class FCLayer:
    def __init__(self, in_shape, out_shape):
        self.weights = np.random.randn(out_shape, in_shape)
        self.bias    = np.random.randn(out_shape, 1)
        self.input   = None

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, input) + self.bias

    def backprop(self, grad, lr):
        out_grad  = np.dot(self.weights.T, grad)
        d_weights = np.dot(grad, self.input.T)
        d_bias    = grad

        self.weights -= lr * d_weights
        self.bias    -= lr * d_bias

        return out_grad
