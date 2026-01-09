import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, f1_score
from src.baseline_preprocessor import build_vocab, create_embedding_matrix, text_to_sequence
from src.model_bilstm import BiLSTMClaimDetector  # We'll create this next

# --- 1. Data Preparation ---
df = pd.read_csv("data/raw/labels_v6.csv")
# Multiclass label: 0=None, 1=Explicit, 2=Implicit
df['multi'] = 0
df.loc[df['explicit'] == 1, 'multi'] = 1
df.loc[(df['claim'] == 1) & (df['explicit'] == 0), 'multi'] = 2

vocab = build_vocab(df['text'])
matrix = create_embedding_matrix("embeddings/crawl-300d-2M.vec", vocab)

# Vectorize for NB/LG (Mean of vectors)
X_mean = np.array(
    [np.mean([matrix[vocab.get(w, 1)] for w in t.lower().split()] or [np.zeros(300)], axis=0) for t in df['text']])
# Vectorize for LSTM (Sequences)
X_seq = np.array([text_to_sequence(t, vocab) for t in df['text']])

# --- 2. Evaluate Naive Bayes & Logistic Regression ---
for task, labels in [("Binary", df['claim']), ("Multiclass", df['multi'])]:
    print(f"\n--- {task} Baselines ---")

    lr = LogisticRegression(max_iter=1000).fit(X_mean, labels)
    print("Logistic Regression:\n", classification_report(labels, lr.predict(X_mean)))

    nb = GaussianNB().fit(X_mean, labels)
    print("Naive Bayes:\n", classification_report(labels, nb.predict(X_mean)))

# --- 3. Evaluate BiLSTM (Overview) ---
# Here you would initialize BiLSTMClaimDetector with 'matrix'
# and run a standard PyTorch training loop.