"""
Step 4: Error Taxonomy Summary

After manual annotation of error cases, this script summarizes
the distribution of linguistic error types.

This provides a high-level view of which discourse phenomena
most frequently challenge claim detection.
"""

import pandas as pd
from collections import Counter

ERROR_PATH = "analysis/error_samples.csv"

df = pd.read_csv(ERROR_PATH)

error_counts = Counter(df["error_type"])

print("\nLinguistic error category distribution:\n")
for category, count in error_counts.most_common():
    print(f"{category}: {count}")