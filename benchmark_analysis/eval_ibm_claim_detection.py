import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score
)

from src.pipeline import ClaimPipeline

"""
Zero-shot evaluation of Reddit-trained claim detector
on the IBM Debater claim sentence benchmark.

Key properties:
- Sentence-level evaluation
- No fine-tuning on IBM data
- Fixed decision threshold (0.5)
- Reports both performance and confidence behavior

This script is intentionally simple and transparent:
it evaluates *what the model already knows*, not how well it can adapt.
"""

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "data/ibm_claims/ibm_claim_sentence_test.csv"
THRESHOLD = 0.5  # sigmoid threshold for binary decision


def main():
    # ---- Load benchmark data ----
    df = pd.read_csv(DATA_PATH)

    # Initialize trained Reddit pipeline
    pipeline = ClaimPipeline()

    gold_labels = []
    pred_labels = []
    confidences = []
    query_types = []

    print(
        f"Running zero-shot evaluation on "
        f"{len(df)} IBM claim sentences...\n"
    )

    # ---- Inference loop ----
    # Note: this is intentionally non-batched to
    # preserve clarity and debuggability
    for _, row in df.iterrows():
        text = str(row["text"])
        gold_label = int(row["gold_claim"])

        pred_label, confidence = pipeline.predict_claim(text)

        gold_labels.append(gold_label)
        pred_labels.append(int(pred_label))
        confidences.append(float(confidence))
        query_types.append(row.get("query_type", "unknown"))

    # -----------------------------
    # Overall evaluation metrics
    # -----------------------------
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_labels,
        pred_labels,
        average="binary",
        zero_division=0
    )
    accuracy = accuracy_score(gold_labels, pred_labels)

    print("IBM Claim Detection (Zero-shot)")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"Accuracy:  {accuracy:.3f}")

    # -----------------------------
    # Confidence behavior analysis
    # -----------------------------
    correct_conf = [
        c for g, p, c in zip(gold_labels, pred_labels, confidences)
        if g == p
    ]
    wrong_conf = [
        c for g, p, c in zip(gold_labels, pred_labels, confidences)
        if g != p
    ]

    print("\nConfidence analysis")
    print(f"Avg confidence (correct): {np.mean(correct_conf):.3f}")
    print(f"Avg confidence (wrong):   {np.mean(wrong_conf):.3f}")

    # -----------------------------
    # Breakdown by query type
    # -----------------------------
    print("\nPerformance by query type")

    df_eval = pd.DataFrame({
        "gold": gold_labels,
        "pred": pred_labels,
        "query_type": query_types
    })

    # Query types reflect different retrieval heuristics
    for qt in sorted(df_eval["query_type"].unique()):
        sub = df_eval[df_eval["query_type"] == qt]

        # Skip extremely small slices to avoid unstable metrics
        if len(sub) < 20:
            continue

        p, r, f, _ = precision_recall_fscore_support(
            sub["gold"],
            sub["pred"],
            average="binary",
            zero_division=0
        )

        print(
            f"{qt:10s} | "
            f"F1: {f:.3f} | "
            f"Precision: {p:.3f} | "
            f"Recall: {r:.3f}"
        )


if __name__ == "__main__":
    main()