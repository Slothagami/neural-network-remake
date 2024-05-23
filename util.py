import tensorflow as tf

def one_hot(length, value):
    return tf.transpose(tf.one_hot([value], length))

def one_hot_accuracy(name, nn, samples, labels):
    correct = 0
    for sample, label in zip(samples, labels):
        label = one_hot(10, label)
        pred  = nn.forward(sample)

        if tf.argmax(pred) == tf.argmax(label):
            correct += 1

    print(f"{name} Accuracy: {correct}/{labels.size} ({correct/labels.size * 100:.2f}%)")
    return correct
