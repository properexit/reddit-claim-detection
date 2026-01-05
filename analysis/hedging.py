"""
Epistemic Hedging Analysis

This module defines a small set of linguistically motivated
epistemic hedging markers and provides utilities for counting
and detecting hedging in predicted claim spans.

Hedging is treated as a surface cue for uncertainty, subjectivity,
or softened commitment to a proposition.
"""

import re


# --------------------------------------------------
# Epistemic hedging markers
# --------------------------------------------------
# These patterns capture common informal hedges used
# in Reddit-style health discussions.

HEDGE_PATTERNS = [
    r"\bi\s+think\b",
    r"\bi\s+feel\b",
    r"\bseems?\b",
    r"\bappears?\b",
    r"\bmight\b",
    r"\bmay\b",
    r"\bcould\b",
    r"\bprobably\b",
    r"\bpossibly\b",
    r"\blikely\b",
    r"\bin\s+my\s+experience\b"
]


def count_hedges(text):
    """
    Count the number of epistemic hedging markers
    appearing in a given text span.

    Args:
        text (str): claim span text

    Returns:
        int: total number of hedging cues found
    """
    text = text.lower()
    count = 0

    for pat in HEDGE_PATTERNS:
        count += len(re.findall(pat, text))

    return count


def has_hedge(text):
    """
    Binary indicator for whether a span contains
    at least one epistemic hedge.

    Args:
        text (str): claim span text

    Returns:
        bool
    """
    return count_hedges(text) > 0