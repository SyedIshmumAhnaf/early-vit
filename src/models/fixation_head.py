# src/models/fixation_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FixationHead(nn.Module):
    """
    Projects a global feature vector (B, D) into a spatial prob map (B, 1, Hm, Wm).
    Simple MLP -> (Hm*Wm), then reshape + log_softmax for KLDivLoss.
    """
    def __init__(self, in_dim=256, hm_size=14, hidden=256):
        super().__init__()
        self.hm_h = hm_size
        self.hm_w = hm_size
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hm_size * hm_size)
        )

    def forward(self, x):  # x: (B, D)
        B = x.size(0)
        logits = self.net(x).view(B, 1, self.hm_h, self.hm_w)  # (B,1,Hm,Wm)
        # log_softmax must take a single dimension; flatten spatial dims, then reshape back
        log_probs = F.log_softmax(logits.view(B, -1), dim=-1).view(B, 1, self.hm_h, self.hm_w)
        return log_probs
