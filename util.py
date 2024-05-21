import numpy as np

def one_hot(length, value):
    vector = np.zeros((length, 1))
    vector[value] = 1
    return vector

def one_hot_accuracy(name, nn, samples, labels):
    correct = 0
    for sample, label in zip(samples, labels):
        label = one_hot(10, label)
        pred  = nn.forward(sample)

        if np.argmax(pred) == np.argmax(label):
            correct += 1

    print(f"{name} Accuracy: {correct}/{labels.size} ({correct/labels.size * 100:.2f}%)")
    return correct
