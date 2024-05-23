import tensorflow as tf

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

    def set_layers(self, layers):
        self.layers = layers

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
    
    def update_batch(self):
        for layer in self.layers:
            # check if layer has update method
            upd = getattr(layer, "update", None)

            # if update function exists
            if callable(upd):
                layer.update()


class FCLayer:
    def __init__(self, in_shape, out_shape):
        self.weights = tf.random.uniform((out_shape, in_shape)) - .5
        self.bias    = tf.random.uniform((out_shape, 1))        - .5
        self.input   = None

        self.weight_deltas = tf.zeros_like(self.weights)
        self.bias_deltas   = tf.zeros_like(self.bias)
        self.n_examples    = 0

    def forward(self, input):
        self.input = input
        return tf.matmul(self.weights, input) + self.bias

    def backprop(self, grad, lr):
        out_grad  = tf.matmul(tf.transpose(self.weights), grad)
        d_weights = tf.matmul(grad, tf.transpose(self.input))
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
        self.weight_deltas = tf.zeros_like(self.weights)
        self.bias_deltas   = tf.zeros_like(self.bias)
        self.n_examples    = 0
