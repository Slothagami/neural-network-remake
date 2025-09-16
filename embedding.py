from tokenizer import make_tokens, word_sequence
from functions import Softmax, CCE
from network   import Network
from util      import one_hot
from layers    import FCLayer
from random import randint
import numpy as np

def window_offset(window_size):
    if randint(0, 1) == 0:
        return randint(1, window_size)
    return -randint(1, window_size)

def one_hot_token(token, tokens):
    return one_hot(len(tokens), tokens.index(token))

def make_embedding_table(dataset_text):
    sequence = word_sequence(dataset_text)
    tokens   = make_tokens(dataset_text)
    n_tokens = len(tokens)

    print(f"Sequence Length: {len(sequence)}")

    nn = Network(CCE(), lr=0.001)
    nn.set_layers([
        FCLayer(n_tokens, 300),
        FCLayer(300, n_tokens),
        Softmax()
    ])

    # train network
    epochs = 1
    batch_size = 64
    errsum = 0
    count  = 0
    for epoch in range(epochs):
        for i, token in enumerate(sequence):
            token_one_hot = one_hot_token(token, tokens)

            # train for every word in context window
            window_size = 3
            for offset in range(-window_size, window_size + 1):
                if (i + offset < 0 or i + offset >= len(sequence)): continue 
                if offset == 0: continue

                target = sequence[i + offset]
                target_one_hot = one_hot_token(target, tokens)

                err = nn.train_sample(token_one_hot, target_one_hot)
                errsum += err
                count  += 1

            if i % (batch_size // 3) == 0 and i != 0: 
                print(f"\tSample {i:05}.")

            if i % batch_size == 0 and i != 0: 
                nn.update_batch()
                print(f"Sample: {i:05}: Error: {errsum / count:.16f}")
                errsum = 0
                count  = 0

    return nn.layers[0].weights

if __name__ == "__main__":
    with open("datasets/rj.txt", "r", encoding="utf-8") as file:
        text = file.read()

    embeddings = make_embedding_table(text)
    np.save("datasets/embedding_table.npy", embeddings)
