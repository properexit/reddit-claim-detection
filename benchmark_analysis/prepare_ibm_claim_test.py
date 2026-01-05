import os
import pandas as pd

"""
Prepare IBM Debater claim sentence test set for zero-shot evaluation.

Context
-------
The original IBM Debater `test_set.csv` file:
- Does NOT include column headers
- Contains multiple metadata fields per row
- Mixes retrieval strategies ("query types")

This script extracts only the sentence text, gold claim label,
and query type to enable a clean, sentence-level claim detection benchmark.

Important:
- No filtering or rebalancing is performed
- Original label distribution is preserved
- This is a *data preparation* step, not modeling
"""

# -----------------------------
# Paths
# -----------------------------
INPUT_PATH = "data/ibm_raw/test_set.csv"
OUTPUT_PATH = "data/ibm_claims/ibm_claim_sentence_test.csv"

# -----------------------------
# Column indices (0-based)
# -----------------------------
# Based on IBM Debater dataset documentation and manual inspection
COL_SENTENCE = 3     # Wikipedia sentence text
COL_QUERY_TYPE = 4   # Retrieval query type (q_mc, q_strict, etc.)
COL_LABEL = 6        # Gold label: 1 = claim, 0 = non-claim


def main():
    # ---- File existence check ----
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    # NOTE:
    # IBM test_set.csv has NO HEADER row
    df = pd.read_csv(INPUT_PATH, header=None)

    # Defensive check against unexpected format changes
    if df.shape[1] < 7:
        raise ValueError(
            f"Unexpected format: expected at least 7 columns, "
            f"got {df.shape[1]}"
        )

    # ---- Extract relevant fields ----
    clean_df = pd.DataFrame({
        "text": df[COL_SENTENCE].astype(str).str.strip(),
        "gold_claim": df[COL_LABEL].astype(int),
        "query_type": df[COL_QUERY_TYPE].astype(str)
    })

    # ---- Sanity checks ----
    # Ensure binary labels only
    if not clean_df["gold_claim"].isin([0, 1]).all():
        raise ValueError(
            "Invalid labels detected: gold_claim must be 0 or 1"
        )

    # ---- Basic dataset statistics ----
    print("Dataset statistics:")
    print(clean_df["gold_claim"].value_counts())

    print("\nQuery type distribution:")
    print(clean_df["query_type"].value_counts())

    # ---- Save cleaned dataset ----
    # We intentionally keep all rows to avoid introducing evaluation bias
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    clean_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved cleaned test set to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()