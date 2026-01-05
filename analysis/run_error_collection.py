"""
Step 4: Linguistic Error Collection

This script identifies mismatches between gold claim annotations and
model predictions in order to support qualitative, discourse-level
error analysis.

The goal is not to fix errors, but to understand the linguistic
phenomena that cause them.
"""

import pandas as pd
from src.pipeline import ClaimPipeline

DATA_PATH = "data/raw/labels_v5.csv"
OUTPUT_PATH = "analysis/error_samples.csv"

pipeline = ClaimPipeline()

df = pd.read_csv(DATA_PATH)

error_records = []

for _, row in df.iterrows():
    text = row["text"]
    gold_label = int(row["claim"])

    prediction = pipeline(text)
    predicted_label = int(prediction.get("claim", False))

    # We only care about disagreements between gold and model
    if gold_label != predicted_label:
        error_records.append({
            "text": text,
            "gold_claim": gold_label,
            "predicted_claim": predicted_label,
            "claim_confidence": prediction.get("claim_confidence", 0.0),
            "predicted_span": prediction.get("span", "")
        })

error_df = pd.DataFrame(error_records)

# Sample a small, manageable subset for manual linguistic annotation
error_df = error_df.sample(
    n=min(40, len(error_df)),
    random_state=42
)

error_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(error_df)} error cases to {OUTPUT_PATH}")