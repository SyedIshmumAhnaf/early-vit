#!/usr/bin/env bash
set -e
python -m pytest -q
python -m src.train --backbone tiny3d --frames 8 --size 64 --train_n 32 --val_n 16 --epochs 1 --batch 4
