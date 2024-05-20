from functions import *
from network   import * 
from util      import *
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf

# load dataset
mnist = tf.keras.datasets.mnist 
(train_batch, train_labels), (test_batch, test_labels) = mnist.load_data()
train_batch = train_batch.reshape(train_batch.shape[0], 28*28, 1).astype("float32") / 255 * 2 - 1
test_batch  = test_batch .reshape(test_batch .shape[0], 28*28, 1).astype("float32") / 255 * 2 - 1

# setup network
nn = Network(MSE(), .01)
nn.config((28*28, 64, 10), FCLayer, Tanh)

# train network
error_plot = []
for i, (sample, label) in enumerate(zip(train_batch, train_labels)):
    label = one_hot(10, label)
    err = nn.train_sample(sample, label)

    if i % 128 == 0:
        nn.update_batch()
        print(f"Error: {err:.8f}")
        error_plot.append(err)


plt.plot(error_plot)
plt.ylim((0,1.5))
plt.ylabel("Training Error")
plt.xlabel("Epoch")
plt.show()
