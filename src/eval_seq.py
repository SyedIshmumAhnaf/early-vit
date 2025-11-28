# src/eval_seq.py
import os, argparse, csv
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from src.models.backbone import build_backbone
from src.models.head import BCEHead
from src.metrics.classification import ap_auc
from src.metrics.mtta import frames_to_seconds
from src.utils.seed import set_seed
from src.data.dada import (
    list_dada_samples_with_keys,
    key_to_coord_path,
    first_event_frame_from_coord,
)

# ---------------- Utils ----------------

def sliding_windows_T(frames_tchw, T):
    """
    frames_tchw: FloatTensor (T,C,H,W) in [0,1]
    Returns list of (C,T,H,W) causal windows (window ends at frame i).
    Pads on the left if needed.
    """
    T_all = frames_tchw.shape[0]
    outs = []
    if T_all >= T:
        for i in range(T_all):
            s = max(0, i - T + 1)
            w = frames_tchw[s:i+1]  # (t',C,H,W)
            if w.shape[0] < T:
                pad = w[0:1].repeat(T - w.shape[0], 1, 1, 1)
                w = torch.cat([pad, w], dim=0)
            outs.append(w.permute(1,0,2,3).contiguous())  # (C,T,H,W)
    else:
        pad = frames_tchw[0:1].repeat(T - T_all, 1, 1, 1)
        padded = torch.cat([pad, frames_tchw], dim=0)  # (T,C,H,W)
        for i in range(padded.shape[0]):
            s = max(0, i - T + 1)
            w = padded[s:i+1]
            if w.shape[0] < T:
                w = torch.cat([w[0:1].repeat(T - w.shape[0], 1, 1, 1), w], dim=0)
            outs.append(w.permute(1,0,2,3).contiguous())
    return outs

def read_video_TCHW(path, H, W):
    """
    Returns float tensor (T,C,H,W) normalized to ImageNet stats.
    """
    frames, _, _ = torchvision.io.read_video(
        path, pts_unit="sec", output_format="TCHW"
    )  # (T,C,H,W) uint8
    frames = frames.float() / 255.0
    frames = torch.nn.functional.interpolate(
        frames.permute(1,0,2,3).unsqueeze(0),  # 1,C,T,H,W
        size=(frames.shape[0], H, W),
        mode="trilinear",
        align_corners=False,
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
    """
    Loads either:
      - structured ckpt: {"bb":..., "cls_head":..., "fix_head":..., "args":..., "epoch":...}
      - flat state_dict saved from nn.Sequential(bb, head)
    """
    bb = build_backbone(backbone_name, out_dim=out_dim).to(device).eval()
    head = BCEHead(out_dim).to(device).eval()
    model = torch.nn.Sequential(bb, head).to(device).eval()

    if ckpt_path and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and "bb" in sd and "cls_head" in sd:
            try:
                bb.load_state_dict(sd["bb"], strict=False)
                head.load_state_dict(sd["cls_head"], strict=False)
            except Exception as e:
                print(f"Warning: structured ckpt load failed ({e}); trying flat load.")
                model.load_state_dict(sd, strict=False)
        else:
            model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("Warning: no checkpoint loaded; evaluating random weights.")
    return model

# --------------- Main -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dada_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="validation",
                    choices=["training","validation","testing"])
    ap.add_argument("--max_items", type=int, default=None)

    ap.add_argument("--backbone", type=str, default="mvitv2_s",
                    choices=["tiny3d","mvitv2_s"])
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")

    ap.add_argument("--batch_windows", type=int, default=16)
    ap.add_argument("--thresholds", type=str, default="0.5")
    ap.add_argument("--score_pool", type=str, default="topkmean",
                    choices=["topkmean","max"])
    ap.add_argument("--k_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dump_csv", type=str, default="eval_summary.csv")
    ap.add_argument("--min_alarm_frame", type=int, default=None)
    args = ap.parse_args()

    effective_min_alarm = args.min_alarm_frame if args.min_alarm_frame is not None else (args.frames - 1)
    print(f"[eval] min_alarm_frame set to {effective_min_alarm} (frames)")


    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_items = args.max_items if args.max_items not in [None, -1] else None
    samples = list_dada_samples_with_keys(args.dada_root, args.split, max_items=max_items)
    print(f"Found {len(samples)} videos for split={args.split}")

    D = 256
    backbone = args.backbone if device=="cuda" or args.backbone=="tiny3d" else "tiny3d"
    model = build_model(backbone, D, args.ckpt, device)

    ths = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]

    all_labels, all_scores = [], []
    rows = []

    for key, path, label in tqdm(samples, desc="Evaluating"):
        frames_tchw = read_video_TCHW(path, args.size, args.size)  # (T,C,H,W)
        wins = sliding_windows_T(frames_tchw, args.frames)
        probs = batched_forward(model, wins, device, batch_size=args.batch_windows)  # (T,)

        # Effective minimum frame where an alarm is allowed
        min_alarm_frame = args.min_alarm_frame if args.min_alarm_frame is not None else (args.frames - 1)

        # Event index from coordinate file (first non-zero fixation)
        event_idx = -1
        if int(label) == 1:
            coord_path = key_to_coord_path(args.dada_root, args.split, key)
            event_idx = first_event_frame_from_coord(coord_path)

        # Per-video score (pool over time)
        if args.score_pool == "max":
            vid_score = float(np.max(probs)) if len(probs) else 0.0
        else:
            k = max(1, int(len(probs) * args.k_frac))
            vid_score = float(np.mean(np.sort(probs)[-k:])) if len(probs) else 0.0

        all_labels.append(int(label))
        all_scores.append(vid_score)

        # First-crossing + absolute mTTA for each threshold (apply min_alarm_frame gating)
        per_th = {}
        for th in ths:
            idx_cross = np.where(probs >= th)[0]
            idx_cross = idx_cross[idx_cross >= min_alarm_frame]  # <-- gating applied here
            first_cross = int(idx_cross[0]) if len(idx_cross) else -1

            abs_mtta = 0
            if int(label) == 1 and event_idx >= 0 and first_cross >= 0 and first_cross <= event_idx:
                abs_mtta = int(event_idx - first_cross)

            per_th[th] = (first_cross, abs_mtta)

        row = {"key": key, "label": int(label), "vid_score": vid_score, "event_idx": event_idx}
        for th in ths:
            fc, mtta = per_th[th]
            row[f"first_cross@{th}"] = fc
            row[f"mtta_frames@{th}"] = mtta
        rows.append(row)


    # classification summary
    ap_score, auc_score = ap_auc(all_scores, all_labels)
    print(f"Video-level Val AP: {ap_score:.3f}  AUC: {auc_score:.3f}")

    # per-threshold mTTA summary
    for th in ths:
        mttas = [r[f"mtta_frames@{th}"] for r in rows if r["label"]==1 and r["event_idx"]>=0]
        if mttas:
            mean_mtta = float(np.mean(mttas))
            print(f"[th={th}] Mean Absolute mTTA: {mean_mtta:.2f} frames (~{frames_to_seconds(mean_mtta):.2f} s)")
        positives = [r for r in rows if r["label"]==1 and r["event_idx"]>=0]
        pre_event_triggers = sum(
            1 for r in positives
            if r[f"first_cross@{th}"] >= 0 and r[f"first_cross@{th}"] <= r["event_idx"]
        )
        print(f"[th={th}] Positives that triggered before event: {pre_event_triggers}/{len(positives) if positives else 0}")

    # CSV dump
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.dump_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote per-video summary to {args.dump_csv}")


if __name__ == "__main__":
    main()
