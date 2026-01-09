import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from src.baseline_preprocessor import build_vocab, create_embedding_matrix, text_to_sequence
from src.model_bilstm import BiLSTMClaimDetector
from src.utils import log_experiment  # Ensure this is imported

# --------------------------------------------------
# 1. Configuration & Data Loading
# --------------------------------------------------
VEC_PATH = "embeddings/crawl-300d-2M.vec"
DATA_PATH = "data/raw/labels_v6.csv"
MAX_LEN = 128
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

df['multi_label'] = 0
df.loc[df['explicit'] == 1, 'multi_label'] = 1
df.loc[(df['claim'] == 1) & (df['explicit'] == 0), 'multi_label'] = 2

# --------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------
vocab = build_vocab(df['text'].astype(str))
embedding_matrix = create_embedding_matrix(VEC_PATH, vocab)


def get_mean_vectors(texts, vocab, matrix):
    vectors = []
    for text in texts:
        idxs = [vocab.get(w.lower(), 1) for w in str(text).split()]
        vectors.append(np.mean(matrix[idxs], axis=0) if idxs else np.zeros(300))
    return np.array(vectors)


X_mean = get_mean_vectors(df['text'], vocab, embedding_matrix)
X_seq = np.array([text_to_sequence(str(t), vocab, MAX_LEN) for t in df['text']])


# --------------------------------------------------
# 3. Evaluation Function for NB and LR
# --------------------------------------------------
def evaluate_sk_models(X, y_bin, y_multi, vocab_size):
    X_train, X_test, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.2, random_state=42)
    _, _, y_train_multi, y_test_multi = train_test_split(X, y_multi, test_size=0.2, random_state=42)

    for name, model in [("Logistic_Regression", LogisticRegression(max_iter=1000)),
                        ("Naive_Bayes", GaussianNB())]:
        print(f"\n--- {name} ---")
        model.fit(X_train, y_train_bin)
        bin_preds = model.predict(X_test)

        model.fit(X_train, y_train_multi)
        multi_preds = model.predict(X_test)

        bin_f1 = f1_score(y_test_bin, bin_preds)
        multi_f1 = f1_score(y_test_multi, multi_preds, average='macro')
        report = classification_report(y_test_multi, multi_preds, target_names=['None', 'Explicit', 'Implicit'])

        results = {"binary_f1": bin_f1, "multiclass_macro_f1": multi_f1}
        config = {"model_type": name, "vocab_size": vocab_size, "embedding": "FastText_Mean"}

        log_experiment(f"{name}_Baseline", config, results, report)


# --------------------------------------------------
# 4. BiLSTM Training Loop
# --------------------------------------------------
def train_bilstm(X, y, num_classes, task_name, embedding_matrix):
    print(f"\nTraining BiLSTM for {task_name}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Corrected data types for loss functions
    y_train_tensor = torch.FloatTensor(y_train) if num_classes == 1 else torch.LongTensor(y_train)
    y_test_tensor = torch.FloatTensor(y_test) if num_classes == 1 else torch.LongTensor(y_test)

    train_ds = TensorDataset(torch.LongTensor(X_train), y_train_tensor)
    test_ds = TensorDataset(torch.LongTensor(X_test), y_test_tensor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = BiLSTMClaimDetector(embedding_matrix, num_classes=num_classes).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    final_loss = 0
    for epoch in range(5):
        model.train()
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            outputs = model(batch_x.to(DEVICE)).squeeze()
            if num_classes == 1:
                preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            else:
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)

    avg_type = 'binary' if num_classes == 1 else 'macro'
    f1 = f1_score(y_test, all_preds, average=avg_type)
    report = classification_report(y_test, all_preds,
                                   target_names=['None', 'Explicit', 'Implicit']) if num_classes > 1 else None

    results = {"f1_score": f1, "final_loss": final_loss}
    config = {"model_type": "BiLSTM", "task": task_name, "epochs": 5, "lr": 1e-3, "hidden_dim": 128}

    log_experiment(f"BiLSTM_{task_name}", config, results, report)


if __name__ == "__main__":
    evaluate_sk_models(X_mean, df['claim'].values, df['multi_label'].values, len(vocab))
    train_bilstm(X_seq, df['claim'].values, 1, "Binary", embedding_matrix)
    train_bilstm(X_seq, df['multi_label'].values, 3, "Multiclass", embedding_matrix)