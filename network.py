import numpy as np
from functions import LossFunction

class Network:
    """
        Manager for a Neural Network Model. Initialized with no layers.
        Call the `network.set_layers()` method to add layers manually. Or use 
        `network.append_layer()`.
    """

    def __init__(self, error_func: LossFunction, lr: float):
        self.layers = []
        self.error_func = error_func
        self.output = None
        self.lr = lr

    def config(self, layers, type, activation):
        for in_size, out_size in zip(layers, layers[1:]):
            self.append_layer(type(in_size, out_size))
            self.append_layer(activation())

    def set_layers(self, layers: list):
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
