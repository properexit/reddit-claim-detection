"""
Input Granularity Ablation Study
--------------------------------

This experiment compares three inference strategies for claim detection,
while keeping the *model itself fixed*.

The goal is to isolate the effect of **input granularity** at inference time.

Variants:
1. Full Post        – classify the entire Reddit post as one unit
2. Any Sentence     – classify each sentence; positive if *any* sentence is a claim
3. Best Sentence    – pipeline strategy (sentence ranking + span-aware reranking)

This is an inference-only ablation:
no retraining, no parameter changes.
"""

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.pipeline import ClaimPipeline, split_sentences_regex


# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "data/raw/labels_v6.csv"
THRESHOLD = 0.5  # kept for conceptual clarity


def evaluate(gold, preds):
    """
    Utility function for consistent metric reporting.
    """
    p, r, f, _ = precision_recall_fscore_support(
        gold,
        preds,
        average="binary",
        zero_division=0
    )
    return round(p, 3), round(r, 3), round(f, 3)


def main():
    # ---- Load dataset ----
    df = pd.read_csv(DATA_PATH)

    texts = df["text"].tolist()
    gold_labels = df["claim"].astype(int).tolist()

    pipeline = ClaimPipeline()

    preds_full_post = []
    preds_any_sentence = []
    preds_best_sentence = []

    # ---- Inference loop ----
    for text in texts:
        # ----------------------------------
        # Variant 1: Full post classification
        # ----------------------------------
        pred_full, _ = pipeline.predict_claim(text)
        preds_full_post.append(int(pred_full))

        # ----------------------------------
        # Variant 2: Any sentence is a claim
        # ----------------------------------
        sentences = split_sentences_regex(text)

        sentence_preds = [
            pipeline.predict_claim(sent)[0]
            for sent in sentences
        ]

        preds_any_sentence.append(int(any(sentence_preds)))

        # ----------------------------------
        # Variant 3: Best sentence (pipeline)
        # ----------------------------------
        result = pipeline.predict_on_long_text(text)
        preds_best_sentence.append(
            int(result.get("claim", False))
        )

    # ---- Compute metrics ----
    p1, r1, f1 = evaluate(gold_labels, preds_full_post)
    p2, r2, f2 = evaluate(gold_labels, preds_any_sentence)
    p3, r3, f3 = evaluate(gold_labels, preds_best_sentence)

    # ---- Report results ----
    print("\nInput Granularity Ablation Results\n")
    print(f"{'Variant':<25} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 55)

    print(f"{'Full Post':<25} {p1:<10} {r1:<10} {f1:<10}")
    print(f"{'Any Sentence':<25} {p2:<10} {r2:<10} {f2:<10}")
    print(f"{'Best Sentence':<25} {p3:<10} {r3:<10} {f3:<10}")


if __name__ == "__main__":
    main()