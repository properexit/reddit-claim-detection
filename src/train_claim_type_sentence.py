"""
Modeling assumption:
Explicitness is often expressed locally at the sentence level rather than
uniformly across an entire Reddit post. To test this hypothesis, we reformulate
claim type classification as a sentence-level task while retaining
document-level supervision.

Empirical observation:
Naively projecting document-level labels to all constituent sentences introduces
substantial label noise and leads to degraded performance. This experiment is
therefore primarily exploratory and not used in the final system.
"""

import os
import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from tqdm import tqdm

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from datasets import SentenceClaimTypeDataset
from model import BertMiniClaimDetector
from utils import set_seed


# --------------------------------------------------
# Training configuration
# --------------------------------------------------
MODEL_NAME = "prajjwal1/bert-mini"

BATCH_SIZE = 32
EPOCHS = 20
LR = 2e-5
SEED = 42
PATIENCE = 3
# --------------------------------------------------


set_seed(SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "labels_v5.csv")
SAVE_DIR = os.path.join(PROJECT_ROOT, "experiments", "claim_type_sentence")
os.makedirs(SAVE_DIR, exist_ok=True)


def train():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device:", device)

    # --------------------------------------------------
    # Dataset construction
    # --------------------------------------------------
    dataset = SentenceClaimTypeDataset(
        csv_path=DATA_PATH,
        tokenizer_name=MODEL_NAME
    )

    labels = np.array(dataset.labels)
    indices = list(range(len(dataset)))

    # Preserve class balance at the sentence level
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=SEED
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------
    # Model and optimization
    # --------------------------------------------------
    model = BertMiniClaimDetector(MODEL_NAME).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    # --------------------------------------------------
    # Class-weighted loss
    # --------------------------------------------------
    # Sentence-level projection introduces a heavy class imbalance.
    # We counteract this using a positive class weight.
    num_explicit = (labels == 1).sum()
    num_implicit = (labels == 0).sum()

    pos_weight = torch.tensor(
        num_implicit / num_explicit,
        device=device,
        dtype=torch.float32
    )

    print(f"Explicit sentences: {num_explicit}")
    print(f"Implicit sentences: {num_implicit}")
    print(f"Using pos_weight: {pos_weight.item():.4f}")

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = 0.0
    epochs_no_improve = 0

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(EPOCHS):
        model.train()
        train_preds, train_labels = [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels_batch)

            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels_batch.cpu().numpy())

        train_f1 = f1_score(train_labels, train_preds, average="macro")

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_batch = batch["label"].to(device)

                logits = model(input_ids, attention_mask)
                preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()

                val_preds.extend(preds)
                val_labels.extend(labels_batch.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average="macro")

        print(
            f"Epoch {epoch + 1} | "
            f"Train Macro-F1: {train_f1:.4f} | "
            f"Val Macro-F1: {val_f1:.4f}"
        )

        # Early stopping based on validation performance
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(
                model.state_dict(),
                os.path.join(
                    SAVE_DIR,
                    "bert_mini_claim_type_sentence_best.pt"
                )
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"\nBest Sentence-Level Val Macro-F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    train()