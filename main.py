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

# setup 
lr = 0.04
nn = Network(MSE(), lr)
nn.config((28*28, 64, 10), FCLayer, Tanh)

batch_size = 128
epochs = 2

print(f"Beginning training: {lr=}, {batch_size=}, {epochs=}\n")

# train network
error_plot = []
for epoch in range(epochs):
    for i, (sample, label) in enumerate(zip(train_batch, train_labels)):
        label = one_hot(10, label)
        err = nn.train_sample(sample, label)

        if i % batch_size == 0:
            nn.update_batch()
            error_plot.append(err)

        if i == train_labels.size-1:
            print(f"Epoch {epoch+1} Error: {err:.8f}")



# test accuracy
correct = 0
for sample, label in zip(train_batch, train_labels):
    label = one_hot(10, label)
    pred  = nn.forward(sample)

    if np.argmax(pred) == np.argmax(label):
        correct += 1

print(f"\nTrain Accuracy: {correct}/{train_labels.size} ({correct/train_labels.size * 100:.2f}%)")

# test accuracy
correct = 0
for sample, label in zip(test_batch, test_labels):
    label = one_hot(10, label)
    pred  = nn.forward(sample)

    if np.argmax(pred) == np.argmax(label):
        correct += 1

print(f"Test  Accuracy: {correct}/{test_labels.size} ({correct/test_labels.size * 100:.2f}%)\n")


plt.plot(error_plot)
plt.ylim((0,1.5))
plt.ylabel("Training Error")
plt.xlabel("Epoch")
plt.show()
