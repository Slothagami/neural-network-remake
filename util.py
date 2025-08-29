import numpy as np

def one_hot(length: int, value: int):
    """
        Returns a one hot column vector with length `length` and value `value`.
        For Example
        ```
        one_hot(3, 0)
        > np.array([[1], 
                    [0], 
                    [0]])
    """
    vector = np.zeros((length, 1))
    vector[value] = 1
    return vector

def one_hot_accuracy(name: str, network, samples, labels, print=True):
    """
        Calculates the accuracy of the model over the data provided and optionally prints the percentage accuracy.
    """
    correct = 0
    for sample, label in zip(samples, labels):
        label = one_hot(10, label)
        pred  = network.forward(sample)

        if np.argmax(pred) == np.argmax(label):
            correct += 1

    if print:
        print(f"{name} Accuracy: {correct}/{labels.size} ({correct/labels.size * 100:.2f}%)")

    return correct
