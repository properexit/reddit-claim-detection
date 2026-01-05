# Reddit Claim Detection for Self-Medication Discussions

This repository implements a **claim detection pipeline** for Reddit posts related to **self-medication and health discussions**.

Given a Reddit post, the system predicts:
1. Whether the post contains a claim  
2. Whether the claim is **explicit** or **implicit**  
3. Which **span of text** expresses the claim  

The project focuses on **practical modeling choices under limited data**, using a lightweight transformer backbone (**BERT-mini**).

---

## Dataset Summary

Each post is annotated with:

| Field | Description |
|------|-------------|
| `text` | Full Reddit post |
| `claim` | Claim present (1/0) |
| `explicit` | Explicit (1) or implicit (0) |
| `explicit_span` | Annotated explicit span |
| `implicit_span` | Annotated implicit span |

### Statistics
- Total posts: **1215**
- Explicit spans: **~67%**
- Implicit spans: **~14%**
- Both spans: **~7%**
- No claim: **~26%**

Span annotations are noisy and are not always exact substrings of the original text.

---

## Task Breakdown

### 1️⃣ Claim Detection (Document-Level)

Binary classification on the full post.

- **Input:** entire Reddit post  
- **Output:** claim probability  
- **Role:** first filter in the pipeline  

---

### 2️⃣ Claim Type Classification (Explicit vs Implicit)

Binary classification applied **only if a claim is detected**.

Two approaches were tested:
- **Document-level classification** (final model)
- **Sentence-level classification** (exploratory)

Sentence-level modeling suffered from **label noise** because claim-type labels were only available at the document level.

---

### 3️⃣ Span Detection

Token-level **BIO tagging**:
- `B-CLAIM`
- `I-CLAIM`
- `O`

Key modeling choices:
- Explicit and implicit spans are **merged into a single span task**
- **Fuzzy matching** is used during training to handle annotation mismatch
- At inference, span extraction is **confidence-based**

---

## Pipeline Overview

The final system follows a **pipeline architecture**:

1. Claim detection  
2. Claim type classification  
3. Span extraction  

### Long Reddit Posts
- Text is split into sentences using a simple regex
- Sentences are ranked by **claim confidence**
- Span detection is applied only to **top candidate sentences**

This avoids extracting spans from irrelevant parts of the post.

---

## Example Usage

```python
from src.pipeline import ClaimPipeline

pipeline = ClaimPipeline()

text = """
I’ve been dealing with chronic pain for years.
Some days are manageable, others are terrible.
Just because an MRI shows a disc bulge doesn’t mean it’s the source of someone’s pain.
Plenty of people have disc bulges and no symptoms at all.
"""

print(pipeline.predict_on_long_text(text))
```

**Output**
```json
{
  "claim": true,
  "claim_confidence": 0.75,
  "claim_type": "implicit",
  "claim_type_confidence": 0.29,
  "span": "Just because an MRI shows a disc bulge doesn’t mean it’s the source of someone’s pain.",
  "source_sentence": "Just because an MRI shows a disc bulge doesn’t mean it’s the source of someone’s pain."
}
```

---

## Results

| Task | Metric | Best Score |
|-----|--------|------------|
| Claim detection | F1 | ~0.88 |
| Claim type (doc-level) | Macro-F1 | ~0.62 |
| Claim type (sentence-level) | Macro-F1 | ~0.50 |
| Span detection | Token Macro-F1 | ~0.83 |

---

## Observations

- Claim detection is comparatively easy at the document level
- Explicit vs implicit classification is challenging due to subtle phrasing
- Sentence-level supervision introduces significant label noise
- Span annotations require fuzzy alignment due to inconsistencies
- A pipeline approach performs more reliably than joint modeling

---

## Code Structure

```
src/
├── datasets.py
├── datasets_span.py
├── model.py
├── model_span.py
├── pipeline.py
├── train.py
├── train_claim_type.py
├── train_claim_type_sentence.py
├── train_span.py
```

---

## Notes

- This is a **project module**, not a benchmark system
- The focus is on **clarity, reproducibility, and realistic modeling choices**
- **Negative results** are retained to document what did not work

---

## What This Project Demonstrates

- Practical claim mining on noisy social media text  
- Handling span annotation mismatch in real datasets  
- Trade-offs between document-level and sentence-level modeling  
- Lightweight transformer pipelines under limited data  
