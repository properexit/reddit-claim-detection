import torch.nn as nn
from transformers import AutoModel


class BertMiniClaimDetector(nn.Module):
    """
    Lightweight binary classifier for claim detection
    and claim-type classification.

    Uses the [CLS] representation from BERT-mini
    followed by a linear projection.
    """

    def __init__(self, model_name):
        super().__init__()

        # Compact encoder to reduce overfitting on small data
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Standard practice: use [CLS] embedding for sequence-level decisions
        cls_embedding = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_embedding)

        return logits.squeeze(-1)