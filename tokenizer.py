import string

def make_tokens(dataset_text, n_tokens):
    """Generates a byte pair encoding based on the text input with the specified number of tokens."""

    tokens  = list(string.printable) # initialize with one of every printable character
    dataset = dataset_text.split(" ") # pretokenize the dataset into words

    while len(tokens) < n_tokens:
        # determine the most frequent token pair
        break
    return tokens

if __name__ == "__main__":
    with open("datasets/rj.txt", "r") as file:
        text = file.read()

    tokens = make_tokens(text, 100)
    print(tokens)
