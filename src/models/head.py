import torch.nn as nn

class BCEHead(nn.Module):
    """Simple binary classifier head."""
    def __init__(self, in_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1)
        )
    def forward(self, x):  # (B, D)
        return self.fc(x).squeeze(-1)  # (B,)
