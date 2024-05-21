# w.ai.fu
AI vtuber made from scratch

DODO (catchup):
- optimize speed
- convolution layers
    - pooling layers

TODO (to go beyond):
- transposed convolution
- diffusion model
- transformer

training notes:
- if the error bottoms out and doesn't go below a threshold, this is probably a sign that the batch size is too high
- using tanh you should normalize the data from -1 to 1 and the weights should be uniformly spread between +-.5
- using sigmoid you should normalize the data from 0 to 1 and the weights should be uniformly spread between +-.5
