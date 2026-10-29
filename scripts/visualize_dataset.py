#!/usr/bin/env python3
"""
visualize_dataset.py
====================
Draw ground-truth boxes on randomly sampled images so annotation quality can be
verified by eye. This is evidence for the thesis, not decoration: reviewers ask
"how do you know your labels are correct?" and these images are the answer.

Outputs into runs/dataset_visualization/:
    <split>_<filename>.jpg   individual annotated samples
    grid_<split>.jpg         contact sheet of the same samples

Usage:
    python scripts/visualize_dataset.py
    python scripts/visualize_dataset.py --split train --count 24 --seed 42
    python scripts/visualize_dataset.py --class-name pothole   # only images containing that class
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "dataset_visualization"

PALETTE = [
    (56, 168, 255), (99, 214, 116), (255, 176, 59), (233, 92, 122),
    (167, 128, 255), (64, 214, 214), (255, 128, 64), (140, 200, 90),
    (210, 120, 200), (120, 190, 255), (250, 220, 90), (150, 150, 240),
]


def load_labels(lpath: Path):
    rows = []
    if not lpath.exists():
        return rows
    for line in lpath.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) == 5:
            try:
                rows.append((int(float(parts[0])), *(float(v) for v in parts[1:])))
            except ValueError:
                pass
    return rows


def draw(img: np.ndarray, rows, names: dict) -> np.ndarray:
    h, w = img.shape[:2]
    for cid, cx, cy, bw, bh in rows:
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        color = PALETTE[cid % len(PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = names.get(cid, str(cid))
        scale = max(0.4, min(0.7, w / 900))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = max(y1, th + 6)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
        cv2.putText(img, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def contact_sheet(paths, cell=320, cols=4):
    rows = (len(paths) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 28, dtype=np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        s = min((cell - 8) / w, (cell - 8) / h)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        y, x = (i // cols) * cell, (i % cols) * cell
        oy, ox = y + (cell - im.shape[0]) // 2, x + (cell - im.shape[1]) // 2
        sheet[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize YOLO ground-truth annotations.")
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--count", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class-name", default=None, help="only sample images containing this class")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    data_path = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    if not data_path.exists():
        print("ERROR: {} not found.".format(data_path))
        return 2
    cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = cfg.get("names") or {}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    base = (data_path.parent / cfg.get("path", ".")).resolve()
    rel = cfg.get(args.split)
    if not rel:
        print("ERROR: split '{}' not declared in data.yaml".format(args.split))
        return 2
    img_dir = (base / rel).resolve()
    lbl_dir = Path(str(img_dir).replace("images", "labels", 1))
    if not img_dir.is_dir():
        print("ERROR: {} does not exist. Run prepare_dataset.py first.".format(img_dir))
        return 2

    imgs = [p for p in sorted(img_dir.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    if not imgs:
        print("ERROR: no images in {}".format(img_dir))
        return 2

    wanted = None
    if args.class_name:
        inv = {v: k for k, v in names.items()}
        if args.class_name not in inv:
            print("ERROR: unknown class '{}'. Known: {}".format(args.class_name, list(inv)))
            return 2
        wanted = inv[args.class_name]
        imgs = [p for p in imgs
                if any(r[0] == wanted for r in load_labels(lbl_dir / (p.stem + ".txt")))]
        print("{:,} images contain class '{}'".format(len(imgs), args.class_name))
        if not imgs:
            return 2

    random.Random(args.seed).shuffle(imgs)
    picks = imgs[:args.count]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, n_boxes, n_empty = [], 0, 0

    for p in picks:
        img = cv2.imread(str(p))
        if img is None:
            print("  skip unreadable: {}".format(p.name))
            continue
        rows = load_labels(lbl_dir / (p.stem + ".txt"))
        if not rows:
            n_empty += 1
        n_boxes += len(rows)
        img = draw(img, rows, names)
        cv2.putText(img, "{}  |  {} box(es)".format(p.name[:40], len(rows)),
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, "{}  |  {} box(es)".format(p.name[:40], len(rows)),
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        dest = out_dir / "{}_{}.jpg".format(args.split, p.stem[:50])
        cv2.imwrite(str(dest), img)
        written.append(dest)

    if written:
        sheet_path = out_dir / "grid_{}.jpg".format(args.split)
        cv2.imwrite(str(sheet_path), contact_sheet(written))
        print("\nWrote {} annotated samples to {}".format(len(written), out_dir))
        print("Contact sheet: {}".format(sheet_path))
        print("Boxes drawn: {:,}   images with zero boxes: {}".format(n_boxes, n_empty))
        if n_empty:
            print("NOTE: images with zero boxes are background/negative samples. A few are")
            print("      healthy; many suggests objects were missed during annotation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
