# Neural Network Remake
This Project is a revamp of my older [Neural Network from Scratch Project](https://github.com/Slothagami/neural-network) with nicer code and slghtly better performance.

This Project impliments a fully connected neural network complete with backpropogation and several loss functions to choose from. This model has achieved 97.28% test accuracy on the MNIST image classification dataset.

For an implimentation of a convolutional neural network, see the older project.

DODO (catchup to old version):
- optimize speed
- convolution layers
    - pooling layers

Future Features:
- transposed convolution
- diffusion model
- transformer

training notes:
- if the error bottoms out and doesn't go below a threshold, this is probably a sign that the batch size is too high
- using tanh you should normalize the data from -1 to 1 and the weights should be uniformly spread between +-.5
- using sigmoid you should normalize the data from 0 to 1 and the weights should be uniformly spread between +-.5
