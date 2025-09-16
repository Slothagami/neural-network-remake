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

def word_sequence(text):
    text  = remove_punctuation(text).lower()
    words = re.split(r"(\s)", text)
    words = [word for word in words if (word != " " and word != "")]
    return words

def make_tokens(text):
    unique = list(set(word_sequence(text)))
    return unique + [" ", "\n"]

if __name__ == "__main__":
    with open("datasets/rj.txt", "r", encoding="utf-8") as file:
        text = file.read()

    tokens = make_tokens(text)
    print(tokens, "\n\n", len(tokens))
