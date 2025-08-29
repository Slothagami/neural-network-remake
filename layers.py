import numpy as np

class FCLayer:
    """
        A Fully connected layer for use inside the `Network` class.
        Will automatically generate initial weights based on `in_shape` and `out_shape`
    """

    def __init__(self, in_shape: int, out_shape: int):
        self.weights = np.random.rand(out_shape, in_shape) - .5
        self.bias    = np.random.rand(out_shape, 1)        - .5
        self.input   = None

        self.weight_deltas = np.zeros_like(self.weights)
        self.bias_deltas   = np.zeros_like(self.bias)
        self.n_examples    = 0

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, input) + self.bias

    def backprop(self, grad, lr):
        out_grad  = np.dot(self.weights.T, grad)
        d_weights = np.dot(grad, self.input.T)
        d_bias    = grad

        # update deltas to change weights after batch
        self.weight_deltas += lr * d_weights
        self.bias_deltas   += lr * d_bias
        self.n_examples += 1

        return out_grad
    
    def update(self):
        # update from the average of the previous samples
        if self.n_examples == 0: return
        self.weights -= self.weight_deltas / self.n_examples
        self.bias    -= self.bias_deltas   / self.n_examples

        # reset deltas
        self.weight_deltas.fill(0)
        self.bias_deltas  .fill(0)
        self.n_examples    = 0
