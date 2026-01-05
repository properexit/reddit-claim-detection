# Reddit Claim Detection & Span Extraction

A lightweight NLP pipeline for identifying medical and self-medication claims in Reddit posts.

---

## Project Overview

This project builds a multi-stage NLP pipeline to detect and analyze claims in Reddit posts related to self-medication and health discussions.

Unlike simple text classification, the system:

1. Detects whether a post contains a claim  
2. Classifies the claim as explicit or implicit  
3. Extracts the textual span where the claim is expressed  
4. Performs optional linguistic analyses (claim typology, hedging, error patterns)

The project emphasizes:

- interpretability  
- linguistic grounding  
- modular design  
- robustness to noisy Reddit text  

The system is implemented using **BERT-mini** for efficiency and reproducibility.

---

## Pipeline Architecture

```mermaid
flowchart TD
    A[Reddit Post] --> B[Sentence Scoring]
    B --> C[Claim Detection]
    C --> D[Claim Type Classification]
    D --> E[Span Extraction (BIO Tagging)]
    E --> F[Linguistic Analysis (Optional)]
```

**Design choices**:

- Span-aware modeling instead of document-only classification  
- Lightweight regex-based sentence splitting (no NLTK dependency)

---

## Repository Structure

```
cv_reddit/
│
├── src/
│   ├── pipeline.py              # End-to-end inference pipeline
│   ├── model.py                 # Claim & claim-type classifier
│   ├── model_span.py            # Span extraction model
│   ├── datasets.py              # Training datasets
│   ├── datasets_span.py         # Span alignment dataset
│   └── utils.py                 # Reproducibility helpers
│
├── analysis/
│   ├── claim_typology.py
│   ├── run_claim_typology_corpus.py
│   ├── hedging.py
│   ├── run_hedging_corpus.py
│   ├── summarize_errors.py
│   └── render_error_examples.py
│
├── experiments/
│   ├── claim_detection/
│   ├── claim_type/
│   ├── span_detection/
│
├── data/
│   └── raw/labels_v5.csv
│
├── run_sample.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tasks Implemented

### Task 1 — Claim Detection

Binary classification:

- Question: Does this post contain a claim?
- Model: BERT-mini + [CLS] classifier
- Metric: F1 score
- Best validation F1: ~0.88

---

### Task 2 — Claim Type Classification

Classifies detected claims as:

- Explicit (clearly asserted)
- Implicit (inferred, contrastive, experiential)

Two variants were explored:

- Document-level (final model)
- Sentence-level (exploratory, noisier)

Best Macro-F1: ~0.62 (document-level)

---

### Task 3 — Span Extraction

Token-level BIO tagging:

- B-CLAIM
- I-CLAIM
- O

Robust to noisy annotations using:

- exact match evaluation
- fuzzy token-level alignment (token F1)

Best validation token Macro-F1: ~0.83

---

## Linguistic Analyses (Optional, Non-Intrusive)

These analyses do not modify the core models.

### Claim Typology (Rule-Based)

Each predicted span is labeled using surface cues:

- causal
- contrastive
- epistemic
- normative

Observed trends:

- Implicit claims are often contrastive or epistemic
- Multi-label claims are common

---

### Hedging Analysis

Counts epistemic hedges (e.g., *might*, *seems*, *I think*).

Findings:

- Explicit claims: ~16% contain hedging
- Implicit claims: ~13% contain hedging
- Explicit spans still hedge surprisingly often

---

### Error Taxonomy

Manual inspection of false positives and negatives revealed:

- narrative framing
- experiential reports
- advice vs assertion
- implicit causality
- contrast without a true claim

Many errors reflect annotation ambiguity rather than clear model failure.

---

## Key Observations

- Implicit claims are linguistically weaker and less confident
- Span-based modeling significantly improves interpretability
- Confidence scores correlate with linguistic clarity
- Many so-called errors stem from unclear annotation boundaries

---

## How to Run

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Train Models

```bash
python src/train.py              # Claim detection
python src/train_claim_type.py   # Claim type classification
python src/train_span.py         # Span extraction
```

### Run Inference

```bash
python run_sample.py
```

Example output:

```json
{
  "claim": true,
  "claim_confidence": 0.71,
  "claim_type": "explicit",
  "claim_type_confidence": 0.56,
  "span": "Just because an MRI shows a disc bulge doesn’t mean it’s the source of pain."
}
```

### Long Text Inference

```python
pipeline.predict_on_long_text(text)
```

Automatically:

- splits long text into sentences
- selects the most claim-like sentence
- extracts the claim span

---

## Hardware & Runtime Notes

- Runs on CPU, CUDA, or Apple MPS
- Trained primarily on Apple M1/M2 (MPS)
- BERT-mini chosen for:
  - fast iteration
  - low memory usage
  - reproducibility

---

## Limitations

- Implicit claim detection remains challenging
- Annotation noise affects span supervision
- Rule-based typology is heuristic, not gold-labeled
- Reddit data contains sensitive health discussions
