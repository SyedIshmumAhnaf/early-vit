import os, argparse, math
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import csv

from src.models.backbone import build_backbone
from src.models.head import BCEHead
from src.metrics.classification import ap_auc
from src.metrics.mtta import frames_to_seconds
from src.data.dada import list_dada_samples
from src.utils.seed import set_seed
from src.data.dada import list_dada_samples_with_keys, key_to_coord_path, first_event_frame_from_coord

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
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)  # broadcast over T
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    frames = (frames - mean) / std
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

    ap.add_argument("--thresholds", type=str, default="0.5")
    ap.add_argument("--score_pool", type=str, default="topkmean", choices=["topkmean","max"])
    ap.add_argument("--k_frac", type=float, default=0.1)
    ap.add_argument("--dump_csv", type=str, default="eval_summary.csv")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # List samples
    samples = list_dada_samples_with_keys(args.dada_root, args.split, max_items=args.max_items)
    print(f"Found {len(samples)} videos for split={args.split}")

    # Build model
    D = 256
    backbone = args.backbone if device=="cuda" or args.backbone=="tiny3d" else "tiny3d"
    model = build_model(backbone, D, args.ckpt, device)

    # Evaluate
    all_labels = []
    all_scores = []  # per-video score (max prob over time)
    ths = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    rows = []

    for key, path, label in tqdm(samples, desc="Evaluating"):
        frames_tchw = read_video_TCHW(path, args.size, args.size)
        wins = sliding_windows_T(frames_tchw, args.frames)
        probs = batched_forward(model, wins, device, batch_size=args.batch_windows)

        #vid_score = float(np.max(probs)) if len(probs) else 0.0
        if args.score_pool == "max":
            vid_score = float(np.max(probs)) if len(probs) else 0.0
        else:
            if len(probs):
                k = max(1, int(len(probs) * args.k_frac))
                vid_score = float(np.mean(np.sort(probs)[-k:]))
            else:
                vid_score = 0.0
        
        all_labels.append(int(label))
        all_scores.append(vid_score)
        event_idx = -1

        if int(label) == 1:
            # absolute mTTA: distance from first alarm to first non-zero coord frame
            coord_path = key_to_coord_path(args.dada_root, args.split, key)
            event_idx = first_event_frame_from_coord(coord_path)  # 0-based; -1 if not found
        
        per_th = {}
        for th in ths:
            idx_cross = np.where(probs >= th)[0]
            first_cross = int(idx_cross[0]) if len(idx_cross) else -1
            abs_mtta = 0
            if int(label) == 1 and event_idx >= 0 and first_cross >= 0 and first_cross <= event_idx:
                abs_mtta = int(event_idx - first_cross)
            per_th[th] = (first_cross, abs_mtta)

        row = {
            "key": key, "label": int(label), "vid_score": vid_score, "event_idx": event_idx,
        }

        for th in ths:
            fc, mtta = per_th[th]
            row[f"first_cross@{th}"] = fc
            row[f"mtta_frames@{th}"] = mtta
        rows.append(row)

    # Classification metrics
    ap_score, auc_score = ap_auc(all_scores, all_labels)
    print(f"Video-level Val AP: {ap_score:.3f}  AUC: {auc_score:.3f}")

    for th in ths:
        mttas = [r[f"mtta_frames@{th}"] for r in rows if r["label"]==1 and r["event_idx"]>=0]
        if mttas:
            mean_mtta = float(np.mean(mttas))
            print(f"[th={th}] Mean Absolute mTTA: {mean_mtta:.2f} frames (~{frames_to_seconds(mean_mtta):.2f} s)")
        positives = [r for r in rows if r["label"]==1 and r["event_idx"]>=0]
        pre_event_triggers = sum(1 for r in positives if r[f"first_cross@{th}"] >= 0 and r[f"first_cross@{th}"] <= r["event_idx"])
        print(f"[th={th}] Positives that triggered before event: {pre_event_triggers}/{len(positives) if positives else 0}")

    # Dump CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.dump_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote per-video summary to {args.dump_csv}")

if __name__ == "__main__":
    main()
