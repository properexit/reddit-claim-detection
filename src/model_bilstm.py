import torch
import torch.nn as nn


class BiLSTMClaimDetector(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, num_classes=1):
        super().__init__()
        num_embeddings, embed_dim = embedding_matrix.shape

        # Load pre-trained FastText weights
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=False  # Allow fine-tuning for better performance
        )

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        # BiLSTM output is hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)

        # Concat the final forward and backward hidden states
        final_h = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        return self.fc(final_h)