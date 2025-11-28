import os
import sys
import argparse

# Allow running the script directly without installing the package.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.dada import list_dada_samples_with_keys


def main():
    ap = argparse.ArgumentParser(description="Quick sanity-check for DADA splits.")
    ap.add_argument("--root", required=True, help="Path to DADA-2000-small root.")
    ap.add_argument("--split", default="validation",
                    choices=["training", "validation", "testing"],
                    help="Split to inspect.")
    ap.add_argument("--max_items", type=int, default=None,
                    help="Optional cap on items (None for all).")
    args = ap.parse_args()

    samples = list_dada_samples_with_keys(args.root, args.split, max_items=args.max_items)
    labels = [int(lbl) for _, _, lbl in samples]
    pos = sum(1 for lbl in labels if lbl == 1)
    neg = sum(1 for lbl in labels if lbl == 0)

    print(f"Total samples: {len(samples)}")
    print(f"Positives: {pos}")
    print(f"Negatives: {neg}")


if __name__ == "__main__":
    main()
