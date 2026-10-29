#!/usr/bin/env python3
"""
prepare_dataset.py
==================
Turn an annotated flat folder into a reproducible YOLO train/val/test split.

Input  (produced by your annotation tool, e.g. CVAT/Label Studio "YOLO export"):
    annotated/
        images/  img001.jpg ...
        labels/  img001.txt ...

Output:
    dataset/images/{train,val,test}/  and  dataset/labels/{train,val,test}/

Thesis-grade guarantees:
  * Fixed RANDOM_SEED -> the split is byte-for-byte reproducible.
  * LEAKAGE GUARD: near-duplicate images are grouped and the whole group is
    assigned to ONE split. Without this, a photo in train and its near-copy in
    test inflates your reported mAP and the result is not defensible.
  * Every image is verified to have a matching label file before splitting.
  * A manifest (reports/split_manifest.csv) records which split each file landed
    in, so the exact experiment can be reconstructed.

Usage:
    python scripts/prepare_dataset.py --src annotated --dst dataset
    python scripts/prepare_dataset.py --src annotated --copy            # copy instead of move
    python scripts/prepare_dataset.py --src annotated --no-leak-guard   # NOT recommended
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

RANDOM_SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
def dhash(path: Path) -> int | None:
    """64-bit perceptual hash used only for the leakage guard."""
    try:
        with Image.open(path) as im:
            g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    except Exception:
        return None
    bits = 0
    for b in (g[:, 1:] > g[:, :-1]).flatten():
        bits = (bits << 1) | int(b)
    return bits


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


class DisjointSet:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def group_near_duplicates(hashes: list[int | None], threshold: int, chunk: int = 128) -> list[int]:
    """Return a group id per image; near-duplicates share a group id."""
    n = len(hashes)
    ds = DisjointSet(n)
    valid = [i for i, h in enumerate(hashes) if h is not None]
    if valid:
        idx = np.array(valid)
        arr = np.array([hashes[i] for i in valid], dtype=np.uint64)
        for s in range(0, len(arr), chunk):
            block = arr[s:s + chunk]
            xor = np.bitwise_xor(block[:, None], arr[None, :])
            dist = _POPCOUNT[xor.view(np.uint8)].reshape(xor.shape + (8,)).sum(axis=2)
            rows, cols = np.where(dist <= threshold)
            for r, c in zip(rows, cols):
                if s + r < c:
                    ds.union(int(idx[s + r]), int(idx[c]))
    return [ds.find(i) for i in range(n)]


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Split an annotated dataset into YOLO train/val/test.")
    ap.add_argument("--src", default="annotated", help="folder with images/ and labels/ subfolders")
    ap.add_argument("--dst", default="dataset", help="output YOLO dataset root")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.20)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--copy", action="store_true", help="copy files instead of moving them")
    ap.add_argument("--no-leak-guard", action="store_true", help="disable near-duplicate grouping")
    ap.add_argument("--dup-threshold", type=int, default=5, help="dhash Hamming distance for 'near-duplicate'")
    ap.add_argument("--allow-empty", action="store_true",
                    help="keep images whose label file is empty (background/negative samples)")
    args = ap.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        print("ERROR: --train/--val/--test must sum to 1.0")
        return 2

    src = (ROOT / args.src) if not Path(args.src).is_absolute() else Path(args.src)
    dst = (ROOT / args.dst) if not Path(args.dst).is_absolute() else Path(args.dst)
    src_img, src_lbl = src / "images", src / "labels"

    if not src_img.is_dir() or not src_lbl.is_dir():
        print("ERROR: expected {} and {} to exist.".format(src_img, src_lbl))
        print("Export your annotations in YOLO format into that layout first.")
        return 2

    images = sorted(p for p in src_img.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print("Found {:,} images in {}".format(len(images), src_img))

    # ---- pair images with labels ---- #
    paired, missing_label, empty_label = [], [], []
    for img in images:
        lbl = src_lbl / (img.stem + ".txt")
        if not lbl.exists():
            missing_label.append(img.name)
            continue
        if lbl.stat().st_size == 0 or not lbl.read_text().strip():
            empty_label.append(img.name)
            if not args.allow_empty:
                continue
        paired.append((img, lbl))

    print("Paired image+label : {:,}".format(len(paired)))
    print("Missing label file : {:,}".format(len(missing_label)))
    print("Empty label file   : {:,}  ({})".format(
        len(empty_label), "kept as background" if args.allow_empty else "EXCLUDED"))
    if missing_label:
        print("  e.g. {}".format(missing_label[:5]))
    if not paired:
        print("ERROR: nothing to split.")
        return 2

    # ---- leakage guard ---- #
    if args.no_leak_guard:
        groups = list(range(len(paired)))
        print("Leakage guard      : DISABLED (not recommended for a thesis)")
    else:
        print("Leakage guard      : hashing {:,} images ...".format(len(paired)))
        hashes = [dhash(img) for img, _ in paired]
        groups = group_near_duplicates(hashes, args.dup_threshold)
        gsize = Counter(groups)
        multi = sum(1 for c in gsize.values() if c > 1)
        print("                     {:,} groups, {:,} contain near-duplicates".format(len(gsize), multi))

    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)

    # ---- split at GROUP level, largest groups first for balance ---- #
    rng = random.Random(args.seed)
    keys = sorted(by_group.keys(), key=lambda k: (-len(by_group[k]), k))
    rng.shuffle(keys)

    n = len(paired)
    targets = {"train": args.train * n, "val": args.val * n, "test": args.test * n}
    assigned = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for k in keys:
        members = by_group[k]
        # give the group to whichever split is furthest below its quota
        split = max(targets, key=lambda s: targets[s] - counts[s])
        assigned[split].extend(members)
        counts[split] += len(members)

    # ---- write out ---- #
    for split in ("train", "val", "test"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    op = shutil.copy2 if args.copy else shutil.move
    manifest_rows = []
    class_counter = defaultdict(Counter)

    for split, idxs in assigned.items():
        for i in sorted(idxs):
            img, lbl = paired[i]
            op(str(img), str(dst / "images" / split / img.name))
            op(str(lbl), str(dst / "labels" / split / lbl.name))
            manifest_rows.append({"file": img.name, "split": split, "group": groups[i]})
            try:
                for line in (dst / "labels" / split / lbl.name).read_text().strip().splitlines():
                    if line.strip():
                        class_counter[split][int(float(line.split()[0]))] += 1
            except Exception:
                pass

    (ROOT / "reports").mkdir(exist_ok=True)
    with open(ROOT / "reports" / "split_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["file", "split", "group"])
        wr.writeheader()
        wr.writerows(sorted(manifest_rows, key=lambda r: (r["split"], r["file"])))

    print("\n" + "=" * 62)
    print("SPLIT RESULT  (seed={}, leakage guard {})".format(
        args.seed, "off" if args.no_leak_guard else "on"))
    print("=" * 62)
    for split in ("train", "val", "test"):
        c = counts[split]
        print("  {:<6} {:>7,} images  ({:5.2f}%)   {:,} objects".format(
            split, c, 100.0 * c / n, sum(class_counter[split].values())))
    print("  {:<6} {:>7,} images".format("TOTAL", n))
    print("\nObjects per class per split:")
    all_ids = sorted({cid for s in class_counter.values() for cid in s})
    print("  {:<10} {:>8} {:>8} {:>8}".format("class_id", "train", "val", "test"))
    for cid in all_ids:
        print("  {:<10} {:>8,} {:>8,} {:>8,}".format(
            cid, class_counter["train"][cid], class_counter["val"][cid], class_counter["test"][cid]))
        if class_counter["val"][cid] == 0 or class_counter["test"][cid] == 0:
            print("     ^ WARNING: class {} is absent from val or test - "
                  "its metrics will be meaningless.".format(cid))

    print("\nSaved manifest: {}".format(ROOT / "reports" / "split_manifest.csv"))
    print("Next: python scripts/validate_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
