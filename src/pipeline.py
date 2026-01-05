import re
import torch
import numpy as np
from transformers import AutoTokenizer

from src.model import BertMiniClaimDetector
from src.model_span import BertMiniSpanTagger


# --------------------------------------------------
# Configuration
# --------------------------------------------------
# We keep all models lightweight for consistency.
MODEL_NAME = "prajjwal1/bert-mini"
MAX_LEN = 512

CLAIM_MODEL_PATH = "experiments/claim_detection/bert_mini_claim_best.pt"
CLAIM_TYPE_MODEL_PATH = "experiments/claim_type/bert_mini_claim_type_best.pt"
SPAN_MODEL_PATH = "experiments/span_detection/bert_mini_span_best.pt"
# --------------------------------------------------


def split_sentences_regex(text):
    """
    Very lightweight sentence splitter.

    We deliberately avoid heavy NLP dependencies here.
    For Reddit-style text, punctuation and newlines
    are usually sufficient.
    """
    parts = re.split(r'(?<=[\.\?\!])\s+|\n+', text)
    return [p.strip() for p in parts if p.strip()]


class ClaimPipeline:
    """
    End-to-end inference pipeline that combines:
      1) Claim detection
      2) Claim type classification (explicit / implicit)
      3) Claim span extraction

    Training is done independently for each component.
    This class only handles inference and integration.
    """

    def __init__(self, device=None):
        # Prefer Apple MPS when available, otherwise fall back gracefully.
        self.device = (
            device
            or ("mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True
        )

        # ---- Claim detection model ----
        self.claim_model = BertMiniClaimDetector(MODEL_NAME).to(self.device)
        self.claim_model.load_state_dict(
            torch.load(CLAIM_MODEL_PATH, map_location=self.device)
        )
        self.claim_model.eval()

        # ---- Claim type model ----
        # Same architecture as claim detection, different weights.
        self.claim_type_model = BertMiniClaimDetector(MODEL_NAME).to(self.device)
        self.claim_type_model.load_state_dict(
            torch.load(CLAIM_TYPE_MODEL_PATH, map_location=self.device)
        )
        self.claim_type_model.eval()

        # ---- Span tagging model ----
        self.span_model = BertMiniSpanTagger(MODEL_NAME).to(self.device)
        self.span_model.load_state_dict(
            torch.load(SPAN_MODEL_PATH, map_location=self.device)
        )
        self.span_model.eval()

    # --------------------------------------------------
    # Claim detection
    # --------------------------------------------------
    def predict_claim(self, text):
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logit = self.claim_model(
                enc["input_ids"],
                enc["attention_mask"]
            )

        prob = torch.sigmoid(logit).item()
        return prob > 0.5, prob

    # --------------------------------------------------
    # Claim type (explicit vs implicit)
    # --------------------------------------------------
    def predict_claim_type(self, text):
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logit = self.claim_type_model(
                enc["input_ids"],
                enc["attention_mask"]
            )

        prob = torch.sigmoid(logit).item()
        label = "explicit" if prob > 0.5 else "implicit"
        return label, prob

    # --------------------------------------------------
    # Span detection (BIO decoding)
    # --------------------------------------------------
    def predict_span(self, text):
        """
        Decode a single claim span from text.

        We rely on BIO structure rather than hard probability
        thresholds, since token-level confidence tends to be
        diffuse for lightweight encoders.
        """
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        offsets = enc["offset_mapping"].squeeze(0).tolist()

        with torch.no_grad():
            logits = self.span_model(input_ids, attention_mask)

        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Label mapping: 0 = O, 1 = I-CLAIM, 2 = B-CLAIM
        B_probs = probs[:, 2]
        I_probs = probs[:, 1]

        # Ignore special tokens ([CLS], [SEP], padding)
        valid_indices = [
            i for i, (s, e) in enumerate(offsets)
            if not (s == 0 and e == 0)
        ]

        if not valid_indices:
            return None

        # Anchor on the strongest B-CLAIM token
        best_b = max(valid_indices, key=lambda i: B_probs[i])

        # Conservative cutoff: if even the best B token is weak,
        # we abstain rather than hallucinate a span.
        if B_probs[best_b] < 0.2:
            return None

        start = best_b
        end = best_b

        # Expand rightward through I-CLAIM tokens
        i = best_b + 1
        while i in valid_indices and I_probs[i] > 0.2:
            end = i
            i += 1

        start_char = offsets[start][0]
        end_char = offsets[end][1]

        if start_char >= end_char:
            return None

        return text[start_char:end_char]

    # --------------------------------------------------
    # Document-level pipeline
    # --------------------------------------------------
    def __call__(self, text):
        """
        Run the full pipeline on a single text span.
        """
        output = {}

        has_claim, claim_prob = self.predict_claim(text)
        output["claim"] = has_claim
        output["claim_confidence"] = round(claim_prob, 3)

        if not has_claim:
            return output

        claim_type, type_prob = self.predict_claim_type(text)
        output["claim_type"] = claim_type
        output["claim_type_confidence"] = round(type_prob, 3)

        span = self.predict_span(text)
        output["span"] = span if span is not None else "NO_SPAN_PREDICTED"

        return output

    # --------------------------------------------------
    # Long-text inference (sentence-aware)
    # --------------------------------------------------
    def predict_on_long_text(self, text, top_k=3):
        """
        For long Reddit posts, claims are often sparse.
        We therefore:
          1) score sentences independently for claim likelihood
          2) re-rank candidates based on span extractability
        """
        sentences = split_sentences_regex(text)

        # Stage 1: recall-oriented filtering
        scored = []
        for sent in sentences:
            _, prob = self.predict_claim(sent)
            scored.append((sent, prob))

        scored.sort(key=lambda x: x[1], reverse=True)
        candidates = scored[:top_k]

        chosen = None
        chosen_score = 0.0

        # Stage 2: precision-oriented re-ranking
        for sent, claim_prob in candidates:
            result = self(sent)

            # Skip sentences with no detected claim or no span
            if not result.get("claim"):
                continue

            if result.get("span") in (None, "NO_SPAN_PREDICTED"):
                continue

            span_len = len(result["span"].split())
            score = claim_prob * span_len

            if score > chosen_score:
                chosen_score = score
                chosen = result
                chosen["source_sentence"] = sent

        if chosen is not None:
            return chosen

        # If nothing looks reliable, abstain explicitly.
        return {
            "claim": False,
            "reason": "no reliable claim sentence detected"
        }