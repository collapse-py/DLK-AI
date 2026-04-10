import torch
import torch.nn as nn

class Brain(nn.Module):
    def __init__(self, vocab_size=50000, d_model=512):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        return x