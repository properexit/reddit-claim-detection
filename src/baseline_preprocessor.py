import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import Counter

def build_vocab(texts, max_vocab=20000):
    words = " ".join(texts).lower().split()
    counts = Counter(words)
    # 0 is for padding, 1 is for unknown words (OOV)
    vocab = {word: i+2 for i, (word, _) in enumerate(counts.most_common(max_vocab))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab

def create_embedding_matrix(vec_path, vocab, embed_dim=300):
    matrix = np.zeros((len(vocab), embed_dim))
    found = 0
    print(f"Extracting vectors from {vec_path}...")
    with open(vec_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f) # Skip header
        for line in tqdm(f):
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            if word in vocab:
                matrix[vocab[word]] = np.array(tokens[1:], dtype='float32')
                found += 1
    print(f"Matched {found}/{len(vocab)} words with FastText vectors.")
    return matrix

def text_to_sequence(text, vocab, max_len=128):
    tokens = text.lower().split()
    seq = [vocab.get(t, 1) for t in tokens[:max_len]]
    return seq + [0] * (max_len - len(seq))