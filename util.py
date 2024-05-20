import numpy as np

def one_hot(length, value):
    vector = np.zeros((length, 1))
    vector[value] = 1
    return vector
