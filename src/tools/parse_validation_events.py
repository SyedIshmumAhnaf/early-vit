import os
import argparse
import csv
from collections import defaultdict

def parse_validation_file(val_txt_path):
    """
    Reads e.g. <DADA_ROOT>/validation/validation.txt

    Expected line format (5 tokens):
        key label frame x y

    Example:
        1/017 1 234 383 383
    """
    if not os.path.isfile(val_txt_path):
        raise FileNotFoundError(f"validation file not found: {val_txt_path}")

    per_key = defaultdict(lambda: {"label": None, "frames": []})

    with open(val_txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 5:
                # skip malformed lines
                continue

            key = parts[0]
            try:
                label = int(parts[1])
                frame_idx = int(parts[2])
                x = int(parts[3])
                y = int(parts[4])
            except ValueError:
                # skip bad numeric parsing
                continue

            rec = per_key[key]
            if rec["label"] is None:
                rec["label"] = label
            else:
                # sanity check: if labels disagree, warn but keep the first
                if rec["label"] != label:
                    print(f"Warning: inconsistent label for key={key}: {rec['label']} vs {label}")

            rec["frames"].append((frame_idx, x, y))

    rows = []
    for key, rec in per_key.items():
        label = rec["label"] if rec["label"] is not None else 0
        frames = sorted(rec["frames"], key=lambda t: t[0])
        # event_frame = first frame with non-zero coords, or -1 if none
        event_frame = -1
        for frame_idx, x, y in frames:
            if x != 0 or y != 0:
                event_frame = frame_idx
                break
        rows.append((key, label, event_frame))

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dada_root", type=str, required=True,
                    help="Path to DADA-2000 root (the folder containing 'validation' etc.)")
    ap.add_argument("--split", type=str, default="validation",
                    choices=["training", "validation", "testing"])
    ap.add_argument("--out_csv", type=str, default="validation_events.csv",
                    help="CSV with columns: key,label,event_frame")
    ap.add_argument("--out_samples", type=str, default="samples_phase4c.txt",
                    help="Text file with lines: key,split for visualization")
    ap.add_argument("--n_pos", type=int, default=6,
                    help="Number of positive clips to sample for phase 4c")
    ap.add_argument("--n_neg", type=int, default=4,
                    help="Number of negative clips to sample for phase 4c")
    args = ap.parse_args()

    val_txt = os.path.join(args.dada_root, args.split, f"{args.split}.txt")
    print(f"[parse] reading split file: {val_txt}")

    rows = parse_validation_file(val_txt)
    print(f"[parse] found {len(rows)} unique clips in {args.split}.txt")

    # Write CSV: key,label,event_frame
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "label", "event_frame"])
        for key, label, event_frame in rows:
            w.writerow([key, label, event_frame])
    print(f"[parse] wrote event summary CSV to {args.out_csv}")

    # Build simple sample list for phase 4c
    positives = [r for r in rows if r[1] == 1 and r[2] >= 0]
    negatives = [r for r in rows if r[1] == 0]

    print(f"[parse] positives with valid event_frame: {len(positives)}")
    print(f"[parse] negatives: {len(negatives)}")

    # Sort positives by event_frame (early accidents first)
    positives_sorted = sorted(positives, key=lambda r: r[2])
    n_pos = min(args.n_pos, len(positives_sorted))
    pos_sel = positives_sorted[:n_pos]

    # Sort negatives by key for determinism
    negatives_sorted = sorted(negatives, key=lambda r: r[0])
    n_neg = min(args.n_neg, len(negatives_sorted))
    neg_sel = negatives_sorted[:n_neg]

    # Build list of (key, split) pairs
    sample_keys = []
    for key, label, event_frame in pos_sel:
        sample_keys.append((key, args.split))
    for key, label, event_frame in neg_sel:
        sample_keys.append((key, args.split))

    # Write samples_phase4c.txt
    with open(args.out_samples, "w", encoding="utf-8") as f:
        for key, split in sample_keys:
            f.write(f"{key},{split}\n")
    print(f"[parse] wrote {len(sample_keys)} samples to {args.out_samples}")
    print("[parse] example samples:")
    for key, split in sample_keys[:5]:
        print(f"  {key},{split}")


if __name__ == "__main__":
    main()
