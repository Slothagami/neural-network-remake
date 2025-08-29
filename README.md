# Neural Network Remake
This Project impliments a fully connected neural network libary for python. This model is capable of achieving 97.28% test accuracy on the MNIST image classification dataset, as shown in the [example code](./main.py).

This Project is a revamp of my older [Neural Network from Scratch Project](https://github.com/Slothagami/neural-network) that implimented a convolutional network in the same manner.

## Features
- Highly customizable training process 
- Construct network layers easily with python arrays.
- Support for fully connected and activation layers.
- Several options for activation functions and loss functions including Categorical Cross Entropy loss (CCE), Mean Squared Error (MSE), Softmax, Tanh, RElU and Sigmoid functions.
- Helper functions for One Hot encoding.

## Installation
After cloning this repository, you can install all dependences by running the command 
```bash
git clone https://github.com/Slothagami/neural-network-remake.git
pip install -r requirements.txt
```

## Usage
You can create a new model using the `Network` class, the network takes a loss function and the learning rate as parameters. You can then call `Network.set_layers()` to construct the model. This function expects an array of Layer objects in the order they appear in the model.

### Training
The function `Network.train_sample()` can be used to send one sample through the network. It expects that training and testing data is a numpy array with shape `(n_samples, sample_size, 1)` and type `float32`. 

This function does not update the model until `Network.update_batch()` is called, witch updates the model with the average gradients from all samples processed since its last call. 

A minimal example of a network with two layers is included below. 

```py
from functions import Sigmoid, CCE, Softmax
from network   import Network, FCLayer
from util      import one_hot, one_hot_accuracy

# load dataset 
train_batch = load_training_data()
test_batch  = load_testing_data()

# set up the network
nn = Network(CCE(), lr=0.9)
nn.set_layers([
    FCLayer(100, 50),
    Sigmoid(),
    FCLayer(50, 10),
    Softmax()
])

# train network for one epoch
for i, (sample, label) in enumerate(zip(train_batch, train_labels)):
    label = one_hot(10, label)
    err   = nn.train_sample(sample, label)
    nn.update_batch()

# print accuracy
one_hot_accuracy("Train", nn, train_batch, train_labels)
correct = one_hot_accuracy("Test", nn, test_batch,  test_labels)
```
