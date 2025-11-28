import os
from pathlib import Path
from dada import _read_split_list  # uses the same parsing logic as the dataset

ROOT = Path("/content/DADA-2000-small")   # change if needed
SPLIT = "validation"

split_txt = ROOT / SPLIT / f"{SPLIT}.txt"
print(f"[INFO] Reading split file: {split_txt}")

items = _read_split_list(str(split_txt))

print(f"[INFO] Total entries in file: {len(items)}")

pos = sum(1 for _, lbl in items if lbl == 1)
neg = sum(1 for _, lbl in items if lbl == 0)

print(f"[INFO] Positives: {pos}")
print(f"[INFO] Negatives: {neg}")

print("\n[HEAD]")
for i, (k, lbl) in enumerate(items[:5]):
    print(f"{i:02d}: {k} {lbl}")

print("\n[TAIL]")
for i, (k, lbl) in enumerate(items[-5:]):
    print(f"{len(items)-5+i:02d}: {k} {lbl}")
