from functions import *
from network   import * 
import numpy as np 
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

# setup network
nn = Network(MSE(), .1)
nn.config((2, 3, 5, 2), FCLayer, Sigmoid)

# run network
target = np.array([[1],[0]])
input  = np.ones((2,1))

# train network
error_plot = []
for epoch in range(200):
    err = nn.train_sample(input, target)
    nn.batch()

    if epoch % 50 == 0:
        print(f"Out: {nn.forward(input).T}, Error: {err}")

    error_plot.append(err)

plt.plot(error_plot)
plt.ylim((0,1))
plt.ylabel("Training Error")
plt.xlabel("Epoch")
plt.show()
