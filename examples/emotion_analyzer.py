"""Example Workload 4: EmotionAnalyzer (NLP Text Sentiment / Emotion Classifier)."""

import torch
import torch.nn as nn
import numpy as np


class EmotionClassifier(nn.Module):
    """Text Emotion Classifier with Embedding, Bidirectional GRU/LSTM, and Linear heads."""

    def __init__(self, vocab_size: int = 5000, embed_dim: int = 64, hidden_dim: int = 128, num_emotions: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_emotions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len)
        embeds = self.embedding(x)
        lstm_out, (hn, _) = self.lstm(embeds)
        # Concatenate forward and backward final states
        feat = torch.cat((hn[-2], hn[-1]), dim=1)
        out = self.classifier(feat)
        return out


def get_model(vocab_size: int = 5000, num_emotions: int = 6) -> nn.Module:
    """Instantiate and return EmotionAnalyzer model."""
    torch.manual_seed(42)
    model = EmotionClassifier(vocab_size=vocab_size, num_emotions=num_emotions)
    model.eval()
    return model


def get_sample_input(batch_size: int = 1, seq_len: int = 32) -> torch.Tensor:
    """Return synthetic tokenized text input IDs."""
    torch.manual_seed(42)
    return torch.randint(1, 4000, (batch_size, seq_len), dtype=torch.int64)


def get_test_data(n_samples: int = 100, seq_len: int = 32, num_emotions: int = 6):
    """Return synthetic tokenized test dataset with emotion labels."""
    np.random.seed(42)
    X = np.random.randint(1, 4000, (n_samples, seq_len)).astype(np.int64)
    y = np.random.randint(0, num_emotions, size=(n_samples,)).astype(np.int64)
    return X, y


if __name__ == "__main__":
    model = get_model()
    x = get_sample_input(4)
    with torch.no_grad():
        out = model(x)
    print("EmotionClassifier loaded. Output shape:", out.shape)
