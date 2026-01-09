import numpy as np
import os
from tqdm import tqdm


def load_fasttext_subset(vec_path, vocab, embed_dim=300):
    """Loads only the vectors for words present in our dataset vocab."""
    word_to_vec = {}
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"FastText file not found at {vec_path}")

    print(f"Loading FastText embeddings from {vec_path}...")
    with open(vec_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        # First line contains count and dim
        next(f)
        for line in tqdm(f):
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            if word in vocab:
                word_to_vec[word] = np.array(tokens[1:], dtype='float32')
    return word_to_vec


def get_mean_vector(text, word_to_vec, embed_dim=300):
    """Converts a string into a single mean vector for LR/NB."""
    words = text.lower().split()
    vectors = [word_to_vec[w] for w in words if w in word_to_vec]
    if not vectors:
        return np.zeros(embed_dim)
    return np.mean(vectors, axis=0)