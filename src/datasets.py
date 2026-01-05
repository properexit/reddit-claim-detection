import re
import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer


# --------------------------------------------------
# Lightweight sentence splitting
# --------------------------------------------------
# We intentionally avoid NLTK or spaCy here.
# For Reddit text, punctuation-based splitting
# is usually sufficient and much more robust.
def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ==================================================
# Task 1: Claim Detection (Document Level)
# ==================================================
class ClaimDetectionDataset(Dataset):
    """
    Dataset for binary claim detection.
    Each instance corresponds to a full Reddit post.
    """

    def __init__(self, csv_path, tokenizer_name, max_len=512):
        self.df = pd.read_csv(csv_path)

        self.texts = self.df["text"].tolist()
        self.labels = self.df["claim"].astype(int).tolist()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


# ==================================================
# Task 2A: Claim Type (Document Level)
# ==================================================
class ClaimTypeDataset(Dataset):
    """
    Document-level claim type classification.
    Only posts that contain claims are included.
    """

    def __init__(self, csv_path, tokenizer_name, max_len=512):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["claim"] == 1].reset_index(drop=True)

        self.texts = self.df["text"].tolist()
        self.labels = self.df["explicit"].astype(int).tolist()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


# ==================================================
# Task 2B: Claim Type (Sentence Level)
# ==================================================
class SentenceClaimTypeDataset(Dataset):
    """
    Sentence-level variant of claim type classification.

    This dataset is not used for final evaluation,
    but is useful for analysis and exploratory training
    when claims are sparse within long posts.
    """

    def __init__(self, csv_path, tokenizer_name, max_len=256):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["claim"] == 1].reset_index(drop=True)

        self.sentences = []
        self.labels = []
        self.doc_ids = []

        # Each sentence inherits the document-level label.
        # This introduces noise, but improves recall in practice.
        for doc_id, row in self.df.iterrows():
            text = row["text"]
            label = int(row["explicit"])

            for sent in split_sentences(text):
                self.sentences.append(sent)
                self.labels.append(label)
                self.doc_ids.append(doc_id)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "doc_id": self.doc_ids[idx]
        }