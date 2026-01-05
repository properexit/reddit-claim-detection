import torch.nn as nn
from transformers import AutoModel


class BertMiniSpanTagger(nn.Module):
    """
    Token-level span tagger built on top of BERT-mini.

    Label scheme:
      0 -> O (outside)
      1 -> I-CLAIM
      2 -> B-CLAIM
    """

    def __init__(self, model_name, num_labels=3):
        super().__init__()

        # Lightweight encoder for efficiency on small datasets
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Simple linear head is sufficient for BIO tagging
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use final hidden states for per-token classification
        token_embeddings = outputs.last_hidden_state
        logits = self.classifier(token_embeddings)

        return logits