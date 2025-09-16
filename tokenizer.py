import string
import re

def byte_pair_encoding(dataset_text, n_tokens):
    """Generates a byte pair encoding based on the text input with the specified number of tokens."""

    tokens  = list(string.printable) # initialize with one of every printable character
    dataset = dataset_text.split(" ") # pretokenize the dataset into words

    while len(tokens) < n_tokens:
        # determine the most frequent token pair
        break
    return list(set(dataset))

def remove_punctuation(text):
    for symbol in string.punctuation:
        text = text.replace(symbol, "")

    return text

def make_tokens(text):
    text = remove_punctuation(text).lower()
    words = re.split(r"\s+", text)
    return list(set(words)) + [" ", "\n"]

if __name__ == "__main__":
    with open("datasets/rj.txt", "r", encoding="utf-8") as file:
        text = file.read()

    tokens = make_tokens(text)
    print(tokens, "\n\n", len(tokens))
