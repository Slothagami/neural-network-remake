import numpy as np

class Network:
    """
        Manager for a Neural Network Model. Initialized with no layers.
        Call the `network.set_layers()` method to add layers manually. Or use 
        `network.append_layer()`.
    """

    def __init__(self, error_func, lr: float):
        self.layers = []
        self.error_func = error_func
        self.output = None
        self.lr = lr

    def config(self, layers, type, activation):
        for in_size, out_size in zip(layers, layers[1:]):
            self.append_layer(type(in_size, out_size))
            self.append_layer(activation())

    def set_layers(self, layers):
        """
            Sets the layers of the network, for example
            ```
            network.set_layers([
                FCLayer(100, 50),
                Sigmoid(),
                FCLayer(50, 10),
                Softmax()
            ])
            ```
        """
        self.layers = layers

    def append_layer(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)

        self.output = output
        return output
    
    def train_sample(self, input, target):
        """
            Passes a single training sample through the model.
            Does not update weights immediately, but instead waits
            for `network.update_batch()` to be called.

            Expects input is a `float32` numpy array with the shape `(sample_size, 1)`.
        """
        self.forward(input) 
        d_error = self.output_error(target)

        for layer in reversed(self.layers):
            d_error = layer.backprop(d_error, self.lr)

        return self.display_error(target)
    
    def output_error(self, target):
        return self.error_func.backprop(target, self.output)
    
    def display_error(self, target):
        return self.error_func.forward(target, self.output)
    
    def update_batch(self):
        """
            Applies the weights obtained from `train_sample()`. 
            Will update the network with the average gradient from every
            training sample processed since the last call to this function.
        """
        for layer in self.layers:
            # check if layer has update method
            upd = getattr(layer, "update", None)

            # if update function exists
            if callable(upd):
                layer.update()


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
