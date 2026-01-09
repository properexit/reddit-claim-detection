"""
Corpus-Level Claim Typology Analysis

This script applies the claim typology to predicted claim spans
across the full dataset in order to examine the distribution
of discourse-level claim types.
"""

import pandas as pd
from src.pipeline import ClaimPipeline
from analysis.claim_typology import analyze_claim_typology

DATA_PATH = "data/raw/labels_v6.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    df = df[df["claim"] == 1].reset_index(drop=True)

    pipeline = ClaimPipeline()

    spans = []

    for text in df["text"].tolist():
        result = pipeline.predict_on_long_text(text)

        if result.get("span") and result["span"] != "NO_SPAN_PREDICTED":
            spans.append(result["span"])

    type_counts, multi_counts = analyze_claim_typology(spans)

    print("\nClaim type distribution:")
    for k, v in type_counts.most_common():
        print(f"{k}: {v}")

    print("\nMulti-label stats:")
    for k, v in multi_counts.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()