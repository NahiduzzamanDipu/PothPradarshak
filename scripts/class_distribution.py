#!/usr/bin/env python3
"""
class_distribution.py
=====================
Dataset statistics + Matplotlib charts for the thesis "Dataset" chapter.

Reports
  * total images / total objects / objects per image
  * objects per class and IMAGES per class (different numbers, both matter)
  * class imbalance ratio and Gini coefficient
  * box-size distribution (YOLO small/medium/large convention)
  * per-split class breakdown, to expose splits that lost a rare class

Charts written to runs/dataset_statistics/:
    class_distribution.png    objects and images per class
    split_distribution.png    stacked per-class counts by split
    box_size_distribution.png box area and aspect-ratio histograms
    objects_per_image.png     how crowded the scenes are

Usage:
    python scripts/class_distribution.py
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "dataset_statistics"
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gini(values) -> float:
    v = np.sort(np.asarray([x for x in values if x >= 0], dtype=float))
    if v.sum() == 0:
        return 0.0
    n = len(v)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def main() -> int:
    ap = argparse.ArgumentParser(description="Dataset class-distribution statistics.")
    ap.add_argument("--data", default="data.yaml")
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

    obj_counts = Counter()                       # boxes per class (all splits)
    img_counts = Counter()                       # images containing class
    split_obj = defaultdict(Counter)
    per_image_objs, areas, aspects = [], [], []
    n_images = 0

    for split in SPLITS:
        rel = cfg.get(split)
        if not rel:
            continue
        img_dir = (base / rel).resolve()
        lbl_dir = Path(str(img_dir).replace("images", "labels", 1))
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            n_images += 1
            lp = lbl_dir / (p.stem + ".txt")
            present, k = set(), 0
            if lp.exists():
                for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    try:
                        cid = int(float(parts[0]))
                        _, _, bw, bh = (float(v) for v in parts[1:])
                    except ValueError:
                        continue
                    obj_counts[cid] += 1
                    split_obj[split][cid] += 1
                    present.add(cid)
                    areas.append(bw * bh)
                    aspects.append(bw / bh if bh > 0 else 0.0)
                    k += 1
            per_image_objs.append(k)
            for cid in present:
                img_counts[cid] += 1

    if not obj_counts:
        print("No labelled objects found. Has the dataset been annotated and prepared?")
        return 2

    total_obj = sum(obj_counts.values())
    ids = sorted(names)
    labels = [names[i] for i in ids]
    ovals = [obj_counts.get(i, 0) for i in ids]
    ivals = [img_counts.get(i, 0) for i in ids]

    # ------------------------------ text ------------------------------- #
    L = []
    A = L.append
    A("=" * 72)
    A("DATASET STATISTICS")
    A("=" * 72)
    A("Total images  : {:,}".format(n_images))
    A("Total objects : {:,}".format(total_obj))
    A("Objects/image : mean {:.2f}, median {:.0f}, max {}".format(
        np.mean(per_image_objs), np.median(per_image_objs), max(per_image_objs)))
    A("Empty images  : {:,} ({:.1f}%)".format(
        per_image_objs.count(0), 100.0 * per_image_objs.count(0) / max(n_images, 1)))
    A("")
    A("{:>4} {:<22} {:>9} {:>8} {:>9} {:>8}".format("ID", "CLASS", "OBJECTS", "SHARE", "IMAGES", "OBJ/IMG"))
    A("-" * 72)
    for i, lab, o, im in zip(ids, labels, ovals, ivals):
        A("{:>4} {:<22} {:>9,} {:>7.2f}% {:>9,} {:>8.2f}".format(
            i, lab, o, 100.0 * o / total_obj, im, o / im if im else 0.0))
    A("-" * 72)
    nz = [v for v in ovals if v > 0]
    zero = [labels[k] for k, v in enumerate(ovals) if v == 0]
    A("Classes declared      : {}".format(len(ids)))
    A("Classes with 0 objects: {} {}".format(len(zero), zero if zero else ""))
    if nz:
        ratio = max(nz) / min(nz)
        A("Imbalance ratio (max/min non-zero): {:.1f}x".format(ratio))
        A("Gini coefficient (0=balanced, 1=extreme): {:.3f}".format(gini(ovals)))
        A("")
        A("IMPACT OF IMBALANCE")
        if ratio > 50:
            A("  SEVERE. Rare classes will be almost never predicted; their per-class")
            A("  recall and mAP will sit near zero and drag mAP@50 down. The model")
            A("  optimises overall loss, so ignoring a class that is 1% of the data is")
            A("  cheap for it. Mitigate by collecting more of the rare classes,")
            A("  oversampling their images, or merging them into a broader class.")
        elif ratio > 10:
            A("  MODERATE. Expect noticeably weaker recall on the rare classes.")
            A("  Report PER-CLASS metrics, not only the overall mAP, or the weakness")
            A("  will be hidden by the well-represented classes.")
        else:
            A("  MILD. Reasonably balanced; overall mAP is a fair summary.")
    A("")
    if areas:
        a = np.array(areas)
        small = int((a < 0.01).sum()); med = int(((a >= 0.01) & (a < 0.09)).sum()); large = int((a >= 0.09).sum())
        A("BOX SIZE (area as fraction of image)")
        A("  small  (<1%)      : {:>7,}  ({:.1f}%)".format(small, 100.0 * small / len(a)))
        A("  medium (1%-9%)    : {:>7,}  ({:.1f}%)".format(med, 100.0 * med / len(a)))
        A("  large  (>9%)      : {:>7,}  ({:.1f}%)".format(large, 100.0 * large / len(a)))
        A("  median box area   : {:.4f} of image".format(float(np.median(a))))
        if small / len(a) > 0.5:
            A("  NOTE: majority small objects - consider training at imgsz 960/1280.")
    A("")
    A("PER-SPLIT OBJECT COUNTS")
    A("{:>4} {:<22} {:>9} {:>9} {:>9}".format("ID", "CLASS", "TRAIN", "VAL", "TEST"))
    A("-" * 72)
    for i, lab in zip(ids, labels):
        t, v, s = split_obj["train"][i], split_obj["val"][i], split_obj["test"][i]
        flag = "  <-- missing from val/test" if (v == 0 or s == 0) and (t > 0) else ""
        A("{:>4} {:<22} {:>9,} {:>9,} {:>9,}{}".format(i, lab, t, v, s, flag))
    A("=" * 72)

    text = "\n".join(L)
    print(text)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_statistics.txt").write_text(text, encoding="utf-8")

    # ----------------------------- charts ------------------------------ #
    order = np.argsort(ovals)[::-1]
    slabels = [labels[i] for i in order]
    sobj = [ovals[i] for i in order]
    simg = [ivals[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(8, len(ids) * 0.7), 5.5))
    x = np.arange(len(ids))
    ax.bar(x - 0.2, sobj, 0.4, label="objects (boxes)", color="#3b82f6")
    ax.bar(x + 0.2, simg, 0.4, label="images containing class", color="#f59e0b")
    ax.set_xticks(x); ax.set_xticklabels(slabels, rotation=45, ha="right")
    ax.set_ylabel("count"); ax.set_title("Class distribution")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for xi, v in zip(x, sobj):
        ax.text(xi - 0.2, v, "{:,}".format(v), ha="center", va="bottom", fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "class_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, len(ids) * 0.7), 5.0))
    bottom = np.zeros(len(ids))
    for split, color in zip(SPLITS, ["#3b82f6", "#10b981", "#ef4444"]):
        vals = np.array([split_obj[split][i] for i in [ids[k] for k in order]], dtype=float)
        ax.bar(x, vals, 0.6, bottom=bottom, label=split, color=color)
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(slabels, rotation=45, ha="right")
    ax.set_ylabel("objects"); ax.set_title("Objects per class per split")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "split_distribution.png", dpi=150); plt.close(fig)

    if areas:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(np.sqrt(areas), bins=50, color="#3b82f6")
        axes[0].set_xlabel("sqrt(box area / image area)"); axes[0].set_ylabel("boxes")
        axes[0].set_title("Relative object size"); axes[0].grid(alpha=0.3)
        axes[1].hist(np.clip(aspects, 0, 5), bins=50, color="#10b981")
        axes[1].set_xlabel("box width / height"); axes[1].set_title("Aspect ratio"); axes[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "box_size_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_image_objs, bins=range(0, max(per_image_objs) + 2), color="#8b5cf6", align="left")
    ax.set_xlabel("objects in image"); ax.set_ylabel("images")
    ax.set_title("Scene crowding"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "objects_per_image.png", dpi=150); plt.close(fig)

    print("\nCharts written to {}".format(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
