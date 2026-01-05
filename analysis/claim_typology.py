"""
Claim Typology Analysis

This module defines a lightweight, linguistically motivated typology
for classifying claim spans based on surface discourse cues.

The goal is not perfect classification, but interpretability:
to approximate how different kinds of claims are realized linguistically
in informal health discourse.
"""

import re
from collections import Counter


# --------------------------------------------------
# Linguistically motivated pattern inventories
# --------------------------------------------------

CAUSAL_PATTERNS = [
    r"\bcauses?\b",
    r"\bleads?\s+to\b",
    r"\bresults?\s+in\b",
    r"\bbecause\b",
    r"\bdue\s+to\b",
    r"\btherefore\b",
    r"\bso\b"
]

CONTRASTIVE_PATTERNS = [
    r"\bbut\b",
    r"\bhowever\b",
    r"\balthough\b",
    r"\bjust\s+because\b",
    r"\bdoesn[’']?t\s+mean\b"
]

NORMATIVE_PATTERNS = [
    r"\byou\s+should\b",
    r"\byou\s+need\s+to\b",
    r"\bone\s+should\b",
    r"\bit\s+is\s+important\s+to\b",
    r"\bmust\b"
]

EPISTEMIC_PATTERNS = [
    r"\bi\s+think\b",
    r"\bi\s+feel\b",
    r"\bseems?\b",
    r"\bmight\b",
    r"\bmay\b",
    r"\bcould\b",
    r"\bprobably\b"
]


PATTERN_MAP = {
    "causal": CAUSAL_PATTERNS,
    "contrastive": CONTRASTIVE_PATTERNS,
    "normative": NORMATIVE_PATTERNS,
    "epistemic": EPISTEMIC_PATTERNS
}


# --------------------------------------------------
# Span-level typology assignment
# --------------------------------------------------

def assign_claim_types(span_text):
    """
    Assign zero or more claim types to a predicted claim span
    based on the presence of discourse-level lexical cues.

    Note: Multiple labels may be assigned to a single span,
    reflecting the overlapping nature of claim types.
    """
    span_text = span_text.lower()
    labels = set()

    for claim_type, patterns in PATTERN_MAP.items():
        for pat in patterns:
            if re.search(pat, span_text):
                labels.add(claim_type)
                break

    return labels


# --------------------------------------------------
# Corpus-level aggregation
# --------------------------------------------------

def analyze_claim_typology(spans):
    """
    Aggregate claim type statistics over a collection of spans.

    Args:
        spans: list of predicted claim spans (strings)

    Returns:
        type_counts: frequency of each claim type
        multi_label_counts: statistics on label overlap
    """
    type_counts = Counter()
    multi_label_counts = Counter()

    for span in spans:
        labels = assign_claim_types(span)

        if not labels:
            multi_label_counts["none"] += 1
            continue

        for lbl in labels:
            type_counts[lbl] += 1

        if len(labels) > 1:
            multi_label_counts["multi_label"] += 1

    return type_counts, multi_label_counts