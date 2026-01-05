"""
Example-Level Claim Typology Demonstration

This script illustrates how the claim typology behaves on a small
set of hand-crafted example sentences, intended for sanity checks
and qualitative demonstration.
"""

from src.pipeline import ClaimPipeline
from analysis.claim_typology import analyze_claim_typology


def main():
    pipeline = ClaimPipeline()

    texts = [
        "Just because an MRI shows a disc bulge doesn’t mean it’s the source of pain.",
        "You should really talk to a doctor before mixing medications.",
        "I think self-medicating helps some people, but it’s risky."
    ]

    spans = []

    for text in texts:
        result = pipeline.predict_on_long_text(text)
        if result.get("span"):
            spans.append(result["span"])

    type_counts, multi_counts = analyze_claim_typology(spans)

    print("Claim type distribution:")
    for k, v in type_counts.items():
        print(f"{k}: {v}")

    print("\nMulti-label stats:")
    for k, v in multi_counts.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()