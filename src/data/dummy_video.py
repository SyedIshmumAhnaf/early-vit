from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from typing import Tuple

class DummyVideoClips(Dataset):
    """
    Generates (B, C=3, T, H, W) random clips + labels + accident time-to-event (tte) metadata.
    Use for CPU smoke tests and pipeline checks. Replace later with real DADA loader.
    """
    def __init__(self, n_samples=64, frames=16, size: Tuple[int,int]=(112,112)):
        self.n = n_samples
        self.T = frames
        self.H, self.W = size
        # 50% positives
        y = np.zeros(n_samples, dtype=np.float32)
        y[: n_samples // 2] = 1.0
        np.random.shuffle(y)
        self.y = y
        # random “time-to-accident” (lower is closer to accident)
        self.tte = np.random.randint(low=1, high=self.T+1, size=n_samples).astype(np.int64)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        x = torch.randn(3, self.T, self.H, self.W)
        y = torch.tensor(self.y[idx])
        tte = torch.tensor(self.tte[idx])  # integer in [1, T]
        return x, y, tte

def make_loader(batch_size=4, n_samples=64, frames=16, size=(112,112), shuffle=True):
    ds = DummyVideoClips(n_samples=n_samples, frames=frames, size=size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)
