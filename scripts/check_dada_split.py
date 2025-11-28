#!/usr/bin/env python3
"""
Sanity-check script for DADA-2000-small splits.

It uses the SAME parsing logic as the dataset:
- reads <root>/<split>/<split>.txt
- uses _read_split_list from dada.py
- counts total / positives / negatives
"""

import argparse
from pathlib import Path
from dada import _read_split_list  # <-- reusing your loader logic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Root of DADA-2000-small (e.g. /content/DADA-2000-small)")
    parser.add_argument("--split", type=str, default="validation",
                        choices=["training", "validation", "testing"])
    args = parser.parse_args()

    root = Path(args.root)
    split = args.split

    split_txt = root / split / f"{split}.txt"
    print(f"[INFO] Reading split file: {split_txt}")

    items = _read_split_list(str(split_txt))

    total = len(items)
    pos = sum(1 for _, lbl in items if lbl == 1)
    neg = sum(1 for _, lbl in items if lbl == 0)

    print(f"[INFO] Total entries: {total}")
    print(f"[INFO] Positives: {pos}")
    print(f"[INFO] Negatives: {neg}")

    # Show a small head/tail preview
    print("\n[HEAD]")
    for i, (k, lbl) in enumerate(items[:5]):
        print(f"{i:02d}: {k} {lbl}")

    print("\n[TAIL]")
    for i, (k, lbl) in enumerate(items[-5:]):
        print(f"{total-5+i:02d}: {k} {lbl}")

if __name__ == "__main__":
    main()
