# Reddit Claim Detection & Span Extraction

A lightweight NLP pipeline for identifying medical and self-medication claims in Reddit posts.

---

## Project Overview

This project builds a **multi-stage NLP pipeline** to detect and analyze claims in Reddit posts related to self-medication and health discussions.

Unlike simple text classification, the system explicitly models **where and how claims are expressed**.

### Pipeline Capabilities
1. Detects whether a post contains a claim  
2. Classifies the claim as **explicit** or **implicit**  
3. Extracts the **textual span** where the claim is expressed  
4. Performs optional **linguistic analyses** (typology, hedging, error patterns)

### Design Emphasis
- Interpretability  
- Linguistic grounding  
- Modular design  
- Robustness to noisy Reddit text  

The system is implemented using **BERT-mini** for efficiency and reproducibility.

---

## Pipeline Architecture

```
Reddit Post
   ↓
Sentence Scoring (claim likelihood)
   ↓
Claim Detection (binary)
   ↓
Claim Type Classification (explicit / implicit)
   ↓
Span Extraction (BIO tagging)
   ↓
Optional Linguistic Analysis
```

### Key Design Choices
- Span-aware modeling instead of document-only classification  
- Sentence scoring used only at inference time (no retraining)  
- Lightweight regex-based sentence splitting (no NLTK dependency)

---

## Tasks Implemented

### Task 1 — Claim Detection

Binary classification task:
- **Question:** Does this text contain a claim?
- **Model:** BERT-mini + `[CLS]` classifier
- **Metric:** F1 score
- **Best validation F1:** ~0.88

---

### Task 2 — Claim Type Classification

Classifies detected claims as:
- **Explicit** — clearly asserted  
- **Implicit** — inferred, contrastive, experiential  

Two modeling variants:
- Document-level (**final model**)  
- Sentence-level (exploratory, noisier supervision)

- **Best Macro-F1 (document-level):** ~0.62

---

### Task 3 — Span Extraction

Token-level BIO tagging:
- `B-CLAIM`
- `I-CLAIM`
- `O`

Training strategy accounts for noisy supervision:
- Exact span matches when available  
- Fuzzy token-level alignment otherwise  

- **Best validation token Macro-F1:** ~0.83

---

## Linguistic Analyses (Optional, Non-Intrusive)

These analyses operate purely on **predicted spans** and do not modify trained models.

---

### Claim Typology (Rule-Based)

Each predicted span is labeled using surface cues:
- Causal  
- Contrastive  
- Epistemic  
- Normative  

**Observed trends:**
- Implicit claims are frequently contrastive or epistemic  
- Multi-label spans are common  
- Many claims combine inference + contrast  

---

### Hedging Analysis

Counts epistemic hedges (e.g., *might*, *seems*, *I think*).

**Corpus-level findings:**
- Explicit claims: ~16.6% contain hedging  
- Implicit claims: ~13.5% contain hedging  
- Explicit claims still hedge surprisingly often  

---

### Error Taxonomy

Manual inspection of false positives and negatives revealed:
- Narrative framing  
- Experiential reports  
- Advice vs assertion  
- Implicit causality  
- Contrast without an actual claim  

Many errors reflect **annotation ambiguity**, not clean model failure.

---

## Input Granularity Ablation

**Question:**  
Should claim detection be performed on the full post or at sentence level?

**Experiment:**  
The same trained model is evaluated under three inference strategies.

| Variant         | Precision | Recall | F1    |
|-----------------|-----------|--------|-------|
| Full Post       | 0.845     | 0.974  | 0.905 |
| Any Sentence    | 0.833     | 0.952  | 0.889 |
| Best Sentence   | 0.833     | 0.952  | 0.889 |

**Observation:**
- Full-post inference yields highest recall and F1  
- Sentence-level inference reduces noise but offers no clear F1 gain  
- Claims in Reddit health posts are often globally distributed  

---

## External Benchmark — IBM Debater Claim Detection

Zero-shot evaluation on the IBM Debater claim sentence benchmark (Wikipedia domain).

### Dataset
- 2,500 Wikipedia sentences  
- 733 claims / 1,767 non-claims  
- Query types: `q_strict`, `q_mc`, `q_that`, `q_cl`  

### Zero-Shot Results

| Metric     | Value |
|------------|-------|
| Precision  | 0.407 |
| Recall     | 0.520 |
| F1         | 0.457 |
| Accuracy   | 0.638 |

**Confidence behavior:**
- Avg confidence (correct): 0.408  
- Avg confidence (wrong): 0.527  

The model is **over-confident on errors**, a known failure mode under domain shift.

### Performance by Query Type

| Query     | Precision | Recall | F1    |
|-----------|-----------|--------|-------|
| q_strict  | 0.548     | 0.518  | 0.533 |
| q_that    | 0.418     | 0.468  | 0.441 |
| q_mc      | 0.280     | 0.574  | 0.377 |
| q_cl      | 0.259     | 0.600  | 0.362 |

**Interpretation:**
- Best performance on strict argumentative claims  
- Degradation on loosely phrased factual/contextual sentences  
- Highlights discourse-level mismatch, not model size alone  

---

## Key Observations
- Implicit claims are linguistically weaker and less confident  
- Span-based modeling improves interpretability  
- Sentence-level inference helps localization but not raw F1  
- Cross-domain transfer remains challenging  
- Many “errors” reflect gray areas in claim definition  

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
python src/train.py
python src/train_claim_type.py
python src/train_span.py
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

### Analysis & Evaluation

```bash
python -m analysis.run_claim_typology_corpus
python -m analysis.run_hedging_corpus
python -m analysis.summarize_errors

python -m ablation_analysis.run_input_granularity_ablation
python -m benchmark_analysis.eval_ibm_claim_detection
```

---

## Hardware & Runtime Notes
- Runs on CPU, CUDA, or Apple MPS  
- Trained primarily on Apple M1/M2  
- BERT-mini chosen for:
  - Fast iteration  
  - Low memory usage  
  - Reproducibility  

---

## Limitations
- Implicit claims remain difficult to define and detect  
- Span supervision is noisy  
- Typology and hedging analyses are heuristic  
- Reddit health data raises ethical considerations  
