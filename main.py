from functions import *
from network   import * 
from util      import *
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf

"""
    Best achieved accuracy:
    dataset=mnist, error=cce, lr=0.95, batch_size=1, epochs=4
    lr=0.95, batch_size=1, epochs=4

    Epoch 1 Error: 0.00268955
    Epoch 2 Error: 0.00264022
    Epoch 3 Error: 0.00172789
    Epoch 4 Error: 0.00056664

    Train Accuracy: 58432/60000 (97.39%)
    Test  Accuracy:  9628/10000 (96.28%)
"""

# load dataset
mnist = tf.keras.datasets.mnist 
(train_batch, train_labels), (test_batch, test_labels) = mnist.load_data()
train_batch = train_batch.reshape(train_batch.shape[0], 28*28, 1).astype("float32") / 255
test_batch  = test_batch .reshape(test_batch .shape[0], 28*28, 1).astype("float32") / 255

# setup 
batch_size = 1
epochs     = 1 #4
lr         = 0.9

nn = Network(CCE(), lr)

nn.set_layers([
    FCLayer(28*28, 100),
    Sigmoid(),
    FCLayer(100, 50),
    Sigmoid(),
    FCLayer(50, 10),
    Softmax()
])


print(f"Beginning training: {lr=}, {batch_size=}, {epochs=}\n")

# train network
error_plot = []
for epoch in range(epochs):
    for i, (sample, label) in enumerate(zip(train_batch, train_labels)):
        label = one_hot(10, label)
        err   = nn.train_sample(sample, label)

        if i % batch_size == 0:
            nn.update_batch()
            error_plot.append(err)

        if i == train_labels.size - 1:
            print(f"Epoch {epoch+1} Error: {err:.8f}")



# test accuracy
one_hot_accuracy("\nTrain", nn, train_batch, train_labels)
correct = one_hot_accuracy("Test ", nn, test_batch,  test_labels)

plt.plot(error_plot)
plt.ylim((0,1))
plt.ylabel("Training Error")
plt.xlabel("Batch")
plt.title(f"Test  Accuracy: {correct}/{test_labels.size} ({correct/test_labels.size * 100:.2f}%) after {epochs} epochs")
plt.show()
