import os, argparse, math
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from src.models.backbone import build_backbone
from src.models.head import BCEHead
from src.metrics.classification import ap_auc
from src.metrics.mtta import mean_relative_mtta, frames_to_seconds
from src.data.dada import list_dada_samples
from src.utils.seed import set_seed

def sliding_windows_T(frames_tchw, T, pad_mode="edge"):
    """
    frames_tchw: torch.FloatTensor (T,C,H,W) in [0,1]
    Returns a list of (C,T,H,W) windows, causal (window ends at frame i).
    If T_all < T, we pad at the start by repeating the first frame.
    """
    T_all = frames_tchw.shape[0]
    if T_all >= T:
        outs = []
        for i in range(T_all):
            s = max(0, i - T + 1)
            w = frames_tchw[s:i+1]  # (t',C,H,W)
            if w.shape[0] < T:
                pad = w[0:1].repeat(T - w.shape[0], 1, 1, 1)
                w = torch.cat([pad, w], dim=0)
            outs.append(w.permute(1,0,2,3).contiguous())  # (C,T,H,W)
        return outs
    else:
        # pad full sequence to T then slide as above
        pad = frames_tchw[0:1].repeat(T - T_all, 1, 1, 1)
        padded = torch.cat([pad, frames_tchw], dim=0)  # (T, C, H, W)
        outs = []
        for i in range(padded.shape[0]):
            s = max(0, i - T + 1)
            w = padded[s:i+1]
            if w.shape[0] < T:
                w = torch.cat([w[0:1].repeat(T - w.shape[0], 1, 1, 1), w], dim=0)
            outs.append(w.permute(1,0,2,3).contiguous())
        return outs

def read_video_TCHW(path, H, W):
    # Returns float tensor (T,C,H,W) in [0,1]
    frames, _, _ = torchvision.io.read_video(path, pts_unit="sec", output_format="TCHW")
    frames = frames.float() / 255.0  # (T,C,H,W) uint8->float
    # Resize spatially to (H,W) without changing T
    frames = torch.nn.functional.interpolate(
        frames.permute(1,0,2,3).unsqueeze(0),  # 1,C,T,H,W
        size=(frames.shape[0], H, W),
        mode="trilinear",
        align_corners=False
    ).squeeze(0).permute(1,0,2,3).contiguous()  # (T,C,H,W)
    return frames

def batched_forward(model, windows, device, batch_size=16):
    probs = []
    for i in range(0, len(windows), batch_size):
        batch = torch.stack(windows[i:i+batch_size], dim=0).to(device)  # (B,C,T,H,W)
        with torch.no_grad():
            logits = model(batch)
            p = torch.sigmoid(logits).detach().cpu().numpy().tolist()
        probs.extend(p)
    return np.asarray(probs, dtype=np.float32)

def build_model(backbone_name, out_dim, ckpt_path, device):
    bb = build_backbone(backbone_name, out_dim=out_dim)
    head = BCEHead(out_dim)
    model = torch.nn.Sequential(bb, head).to(device).eval()
    if ckpt_path and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("Warning: no checkpoint loaded; evaluating random weights.")
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dada_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="validation", choices=["training","validation","testing"])
    ap.add_argument("--max_items", type=int, default=64)

    ap.add_argument("--backbone", type=str, default="mvitv2_s", choices=["tiny3d","mvitv2_s"])
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--ckpt", type=str, default="checkpoints/last.pt")

    ap.add_argument("--batch_windows", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # List samples
    samples = list_dada_samples(args.dada_root, args.split, max_items=args.max_items)
    print(f"Found {len(samples)} videos for split={args.split}")

    # Build model
    D = 256
    backbone = args.backbone if device=="cuda" or args.backbone=="tiny3d" else "tiny3d"
    model = build_model(backbone, D, args.ckpt, device)

    # Evaluate
    all_labels = []
    all_scores = []  # per-video score (max prob over time)
    pos_seqs = []    # probability sequences for positives (for mTTA)
    fps_assumed = 30.0  # used for seconds conversion if needed

    for path, label in tqdm(samples, desc="Evaluating"):
        # Read full video and slide windows
        frames_tchw = read_video_TCHW(path, args.size, args.size)  # (T,C,H,W)
        wins = sliding_windows_T(frames_tchw, args.frames)

        # Batched forward
        # Construct (B,C,T,H,W) batch by stacking wins
        probs = batched_forward(model, wins, device, batch_size=args.batch_windows)  # (T,) approx
        # Per-video score for AP/AUC: we can use max over time (or mean); max is common for “did accident occur?”
        vid_score = float(np.max(probs)) if len(probs) else 0.0
        all_labels.append(int(label))
        all_scores.append(vid_score)

        if int(label) == 1:
            pos_seqs.append(probs)

    # Classification metrics
    ap_score, auc_score = ap_auc(all_scores, all_labels)
    print(f"Video-level Val AP: {ap_score:.3f}  AUC: {auc_score:.3f}")

    # Relative mTTA (positives only) in frames and seconds
    rel_mtta_frames = mean_relative_mtta(pos_seqs, thresh=args.threshold)
    rel_mtta_secs = frames_to_seconds(rel_mtta_frames, fps=fps_assumed)
    print(f"Mean Relative mTTA: {rel_mtta_frames:.2f} frames  (~{rel_mtta_secs:.2f} s) at threshold={args.threshold}")

if __name__ == "__main__":
    main()
