import os
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF
import random
import re

# --------- helpers ---------
def _resolve_video_path(root_rgb: str, key: str) -> Optional[str]:
    """key like '1/022' -> try mp4/avi/mkv under rgb_videos/1/022.*"""
    base = os.path.join(root_rgb, key.replace("\\", "/"))
    for ext in (".mp4", ".avi", ".mkv", ".webm", ".mov"):
        p = base + ext
        if os.path.isfile(p):
            return p
    # Sometimes files are saved without zero padding; try loose search
    folder = os.path.dirname(base)
    stem = os.path.basename(base)
    if os.path.isdir(folder):
        for fn in os.listdir(folder):
            name, ext = os.path.splitext(fn)
            if name == stem and ext.lower() in (".mp4",".avi",".mkv",".webm",".mov"):
                return os.path.join(folder, fn)
    return None

def _read_split_list(split_txt: str) -> List[Tuple[str,int]]:
    """
    Each line example (from your snippet):
       '1/022 1 15 164 137'
    We'll parse 'key label ...' and ignore the rest for v1.
    """
    items = []
    with open(split_txt, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"): 
                continue
            parts = ln.split()
            if len(parts) < 2:
                continue
            key = parts[0].strip()
            try:
                label = int(parts[1])
            except Exception:
                # be safe: skip bad lines
                continue
            items.append((key, label))
    return items

def _temporal_sample_indices(num_frames: int, T: int) -> List[int]:
    """uniform sample T frame indices from [0, num_frames-1]; loop-pad if needed"""
    if num_frames <= 0:
        return [0]*T
    if num_frames >= T:
        # uniform linspace
        idx = torch.linspace(0, num_frames-1, steps=T).round().long().tolist()
        return idx
    else:
        # loop-pad
        base = list(range(num_frames))
        out = []
        i = 0
        while len(out) < T:
            out.append(base[i % num_frames])
            i += 1
        return out

# --------- dataset ---------
class DADAClips(Dataset):
    """
    Yields (clip:FloatTensor [3,T,H,W], label:0/1 tensor, tte:int frames).
    Provisional: accident assumed at the end of the clip if label==1 (tte=T).
    """
    def __init__(self, 
                 root_dir: str,          # e.g., '/content/DADA-2000-small'
                 split: str = "training",# 'training'|'validation'|'testing'
                 frames: int = 16,
                 size: Tuple[int,int] = (112,112),
                 max_items: Optional[int] = None):
        super().__init__()
        self.root = root_dir
        self.split = split
        self.frames = frames
        self.H, self.W = size
        self.rgb_root = os.path.join(self.root, split, "rgb_videos")
        split_txt = os.path.join(self.root, split, f"{split}.txt")
        items = _read_split_list(split_txt)

        # Resolve paths and keep those that exist
        self.samples = []
        for key, label in items:
            p = _resolve_video_path(self.rgb_root, key)
            if p is not None:
                self.samples.append((p, label))
        if max_items is not None:
            self.samples = self.samples[:max(0, int(max_items))]

        if len(self.samples) == 0:
            raise RuntimeError(f"No videos found for split={split} under {self.rgb_root}. "
                               f"Check file structure and extensions.")

    def __len__(self): 
        return len(self.samples)

    def _resize_crop(self, frame: torch.Tensor) -> torch.Tensor:
        # frame: (H,W,C) uint8 -> to PIL-less transforms for speed
        # convert to CHW float [0,1], resize, center-crop to exactly self.Hxself.W
        img = frame.permute(2,0,1).float() / 255.0  # C,H,W
        img = torchvision.transforms.functional.resize(img, [self.H, self.W], antialias=True)
        # already exact size due to resize; add normalize later if needed
        return img

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # torchvision.io.read_video returns (T,H,W,C) in uint8 by default
        # We disable audio for speed.
        frames, _, _ = torchvision.io.read_video(path, pts_unit="sec", output_format="TCHW")  # (T,C,H,W)
        # Depending on torchvision version, output_format="TCHW" may already return T,C,H,W
        if frames.dim() == 4 and frames.shape[1] in (1,3):
            # already (T,C,H,W)
            T_all = frames.shape[0]
            idxs = _temporal_sample_indices(T_all, self.frames)
            clip = frames[idxs]  # (T,C,H,W)
            # resize to (C,T,H,W)
            clip = clip.float() / 255.0
            # per-frame resize (vectorized)
            clip = torch.nn.functional.interpolate(
                clip.permute(1,0,2,3).unsqueeze(0),  # 1,C,T,H,W -> treat T as temporal dim
                size=(self.frames, self.H, self.W),
                mode="trilinear",
                align_corners=False
            ).squeeze(0)  # C,T,H,W
        else:
            # Fallback path: use (T,H,W,C)
            frames_hwcz, _, _ = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")
            T_all = frames_hwcz.shape[0]
            idxs = _temporal_sample_indices(T_all, self.frames)
            sel = frames_hwcz[idxs]  # (T,H,W,C)
            # resize each frame then stack
            imgs = [self._resize_crop(sel[t]) for t in range(sel.shape[0])]  # list of (C,H,W)
            clip = torch.stack(imgs, dim=1)  # (C,T,H,W)

        if self.split == "training":
            # Random spatial crop (keep 88–100% of area), then resize back
            scale = random.uniform(0.88, 1.0)
            newH, newW = int(self.H*scale), int(self.W*scale)
            clip = torch.nn.functional.interpolate(
                clip.unsqueeze(0), size=(self.frames, newH, newW),
                mode="trilinear", align_corners=False
            ).squeeze(0)
            # center-crop back to (H,W)
            top = max(0, (newH - self.H)//2); left = max(0, (newW - self.W)//2)
            clip = clip[:, :, top:top+self.H, left:left+self.W]

            # Light brightness/contrast jitter (±10%), *consistent across frames*
            b = 1.0 + random.uniform(-0.1, 0.1)
            c = 1.0 + random.uniform(-0.1, 0.1)
            clip = torch.clamp((clip * c + (b - 1.0)), -3.0, 3.0)  # clamp pre-normalization range a bit

        y = torch.tensor(label, dtype=torch.float32)
        tte = torch.tensor(self.frames, dtype=torch.int64)  # provisional; refine later with true accident frame
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1,1)
        clip = (clip - mean) / std
        return clip, y, tte

def make_dada_loader(root_dir: str, split: str, batch: int = 4, frames: int = 16,
                     size: Tuple[int,int]=(112,112), shuffle=True, max_items: Optional[int]=None):
    ds = DADAClips(root_dir=root_dir, split=split, frames=frames, size=size, max_items=max_items)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=2, pin_memory=True, drop_last=False)

def list_dada_samples(root_dir: str, split: str, max_items: Optional[int] = None):
    """
    Returns list of (video_path, label) for the split, resolved via rgb_videos.
    """
    rgb_root = os.path.join(root_dir, split, "rgb_videos")
    split_txt = os.path.join(root_dir, split, f"{split}.txt")
    items = _read_split_list(split_txt)
    samples = []
    for key, label in items:
        p = _resolve_video_path(rgb_root, key)
        if p is not None:
            samples.append((p, label))
    if max_items is not None and max_items > 0:
        samples = samples[:max_items]
    return samples

def key_to_coord_path(root_dir: str, split: str, key: str) -> str:
    # key '1/022' -> '<root>/<split>/coordinate/1/022_coordinate.txt'
    scene, clip = key.split("/")
    return os.path.join(root_dir, split, "coordinate", scene, f"{clip}_coordinate.txt")

def first_event_frame_from_coord(coord_path: str) -> int:
    """
    Returns 0-based frame index of first non-zero fixation (x,y), or -1 if none.
    """
    if not os.path.isfile(coord_path):
        return -1
    with open(coord_path, "r", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            ln = ln.strip()
            if not ln: 
                continue
            # expect "x,y"
            try:
                x_str, y_str = ln.split(",")
                x = int(x_str); y = int(y_str)
                if x != 0 or y != 0:
                    return i
            except Exception:
                continue
    return -1

def list_dada_samples_with_keys(root_dir: str, split: str, max_items: int | None = None):
    """
    Returns list of (key, video_path, label).
    """
    rgb_root = os.path.join(root_dir, split, "rgb_videos")
    split_txt = os.path.join(root_dir, split, f"{split}.txt")
    items = _read_split_list(split_txt)
    samples = []
    for key, label in items:
        p = _resolve_video_path(rgb_root, key)
        if p is not None:
            samples.append((key, p, label))
    if max_items:
        samples = samples[:max_items]
    return samples
