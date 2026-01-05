"""
Step 5: Rendering Qualitative Error Examples

This script converts annotated qualitative error examples into
a human-readable Markdown document suitable for reports, slides,
or repository documentation.
"""

import pandas as pd
from pathlib import Path

CSV_PATH = Path("analysis/error_samples.csv")
OUT_PATH = Path("analysis/error_examples.md")
MAX_CHARS = 500


def render_markdown():
    df = pd.read_csv(CSV_PATH)

    output = []
    output.append("# Qualitative Error Analysis\n")
    output.append(
        "This section presents representative error cases illustrating "
        "linguistic and discourse phenomena that challenge automated "
        "claim detection in self-medication discussions.\n"
    )

    for idx, row in df.iterrows():
        output.append(f"## Example {idx + 1}\n")

        text = str(row["text"])
        excerpt = text[:MAX_CHARS] + ("..." if len(text) > MAX_CHARS else "")

        output.append("**Text excerpt**\n")
        output.append(f"> {excerpt}\n")

        output.append("**Model behavior**\n")
        output.append(f"- Gold label: `{int(row['gold_claim'])}`\n")
        output.append(f"- Predicted label: `{int(row['predicted_claim'])}`\n")
        output.append(f"- Confidence: `{row['claim_confidence']:.3f}`\n")

        span = row.get("predicted_span", "")
        if isinstance(span, str) and span.strip():
            output.append(f"- Predicted span: _{span}_\n")

        output.append("\n**Linguistic category**\n")
        output.append(f"- `{row['error_type']}`\n")

        explanation = row.get("linguistic_explanation", "")
        if isinstance(explanation, str) and explanation.strip():
            output.append("\n**Explanation**\n")
            output.append(f"{explanation}\n")

        output.append("\n---\n")

    OUT_PATH.write_text("\n".join(output), encoding="utf-8")
    print(f"Rendered qualitative analysis to {OUT_PATH}")


if __name__ == "__main__":
    render_markdown()