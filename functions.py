import tensorflow as tf

class Sigmoid:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return 1 / (1 + tf.exp(-x))
    
    def backprop(self, d_err, lr):
        return d_err * self.forward(self.input) * (1 - self.forward(self.input))
    
class Tanh:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return tf.tanh(x)
    
    def backprop(self, d_err, lr):
        return d_err * ( 1 - tf.square(tf.tanh(self.input)) )
    
class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return tf.reduce_max(0, x)
    
    def backprop(self, d_err, lr):
        return d_err * float(self.input > 0)
    
class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, x):
        x  -= tf.reduce_max(x) # shift input to avoid overflows
        exp = tf.exp(x)
        self.output = exp / tf.reduce_sum(exp)
        return self.output
    
    def backprop(self, d_err, lr):
        # can be optimized with a different formula/methods.
        stack_size = tf.size(self.output).numpy()
        y_mat = tf.tile(self.output, tf.constant([1, stack_size]))

        derivative = y_mat * (tf.eye(stack_size) - tf.transpose(y_mat))

        return tf.matmul(derivative, d_err)
    
class MSE:
    def forward(self, target, pred):
        return tf.reduce_mean(tf.square(target - pred))
    
    def backprop(self, target, pred):
        return -2 * (target - pred) / tf.size(target, tf.float32)

EPS = 1e-16
class CCE: # Categorical Cross Entropy
    def forward(self, target, pred):
        return -tf.reduce_sum(target * tf.math.log(pred + EPS)) / tf.size(target, tf.float32)
    
    def backprop(self, target, pred):
        return (pred - target) / tf.size(target, tf.float32)
