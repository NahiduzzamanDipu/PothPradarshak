#!/usr/bin/env python3
"""
validate_dataset.py
===================
Hard validation of a YOLO dataset. Nothing is silently ignored: every problem is
counted, listed, and the script exits non-zero if the dataset is not trainable.

Checks
  * images without labels / labels without images
  * empty label files
  * malformed lines (wrong field count, non-numeric)
  * class ids outside the range declared in data.yaml
  * coordinates outside [0,1]  (YOLO format is normalised)
  * negative or zero width/height
  * boxes whose extent runs off the image edge (cx +- w/2 outside [0,1])
  * degenerate boxes (< 2 px on a side at the training resolution)
  * corrupted / unreadable images
  * unsupported image formats
  * duplicate filename stems inside a split
  * the SAME image appearing in more than one split (data leakage)

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --data data.yaml --fix-clip
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parent.parent
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


class Report:
    def __init__(self) -> None:
        self.counts: Counter = Counter()
        self.details: dict[str, list[str]] = defaultdict(list)

    def add(self, key: str, detail: str | None = None, limit: int = 15) -> None:
        self.counts[key] += 1
        if detail and len(self.details[key]) < limit:
            self.details[key].append(detail)

    def n(self, key: str) -> int:
        return self.counts[key]


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a YOLO dataset.")
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--imgsz", type=int, default=640, help="training size, used for degenerate-box check")
    ap.add_argument("--fix-clip", action="store_true",
                    help="clip slightly out-of-range coordinates back into [0,1] and rewrite the label")
    args = ap.parse_args()

    data_path = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    if not data_path.exists():
        print("ERROR: {} not found. Create it first.".format(data_path))
        return 2

    cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = cfg.get("names") or {}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    nc = len(names)
    base = (data_path.parent / cfg.get("path", ".")).resolve()

    print("data.yaml : {}".format(data_path))
    print("dataset   : {}".format(base))
    print("classes   : {} -> {}".format(nc, list(names.values())[:12]))
    print()

    rep = Report()
    per_split: dict[str, dict] = {}
    stem_to_splits: dict[str, list[str]] = defaultdict(list)
    class_counts: Counter = Counter()
    total_boxes = 0

    for split in SPLITS:
        rel = cfg.get(split)
        if not rel:
            print("[{}] not declared in data.yaml - skipped".format(split))
            continue
        img_dir = (base / rel).resolve()
        lbl_dir = Path(str(img_dir).replace("images", "labels", 1)) \
            if "images" in str(img_dir) else img_dir.parent / "labels"
        if not img_dir.is_dir():
            print("[{}] MISSING image folder: {}".format(split, img_dir))
            rep.add("missing_split_folder", str(img_dir))
            continue

        imgs = [p for p in sorted(img_dir.iterdir()) if p.is_file()]
        lbls = [p for p in sorted(lbl_dir.iterdir()) if p.suffix == ".txt"] if lbl_dir.is_dir() else []
        img_stems, seen_stems = {}, Counter()

        for p in imgs:
            if p.suffix.lower() not in SUPPORTED:
                rep.add("unsupported_format", "{}/{}".format(split, p.name))
                continue
            seen_stems[p.stem] += 1
            img_stems[p.stem] = p
            stem_to_splits[p.stem].append(split)

        for stem, c in seen_stems.items():
            if c > 1:
                rep.add("duplicate_stem_in_split", "{}/{}".format(split, stem))

        n_boxes_split = 0
        for stem, ipath in img_stems.items():
            # ---- image integrity ---- #
            try:
                with Image.open(ipath) as im:
                    im.load()
                    iw, ih = im.size
                if iw < 2 or ih < 2:
                    rep.add("degenerate_image", "{}/{} {}x{}".format(split, ipath.name, iw, ih))
            except Exception as exc:
                rep.add("corrupted_image", "{}/{}: {}".format(split, ipath.name, exc))
                continue

            # ---- label ---- #
            lpath = lbl_dir / (stem + ".txt")
            if not lpath.exists():
                rep.add("missing_label", "{}/{}".format(split, ipath.name))
                continue
            raw = lpath.read_text(encoding="utf-8", errors="ignore").strip()
            if not raw:
                rep.add("empty_label", "{}/{}".format(split, lpath.name))
                continue

            new_lines, changed = [], False
            for ln, line in enumerate(raw.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                where = "{}/{}:{}".format(split, lpath.name, ln)

                if len(parts) != 5:
                    rep.add("malformed_line", "{} -> {} fields".format(where, len(parts)))
                    continue
                try:
                    cid = int(float(parts[0]))
                    cx, cy, bw, bh = (float(v) for v in parts[1:])
                except ValueError:
                    rep.add("non_numeric", where)
                    continue

                bad = False
                if cid < 0 or cid >= nc:
                    rep.add("invalid_class_id", "{} -> class {} (valid 0..{})".format(where, cid, nc - 1))
                    bad = True
                if bw <= 0 or bh <= 0:
                    rep.add("nonpositive_wh", "{} -> w={} h={}".format(where, bw, bh))
                    bad = True
                if any(v < 0 for v in (cx, cy, bw, bh)):
                    rep.add("negative_value", where)
                    bad = True
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0) or bw > 1.0 or bh > 1.0:
                    rep.add("coord_out_of_range", "{} -> cx={} cy={} w={} h={}".format(where, cx, cy, bw, bh))
                    bad = True

                x1, y1 = cx - bw / 2, cy - bh / 2
                x2, y2 = cx + bw / 2, cy + bh / 2
                if x1 < -1e-6 or y1 < -1e-6 or x2 > 1 + 1e-6 or y2 > 1 + 1e-6:
                    rep.add("box_outside_image", "{} -> [{:.3f},{:.3f},{:.3f},{:.3f}]".format(
                        where, x1, y1, x2, y2))
                    if args.fix_clip and not bad:
                        x1, y1 = max(0.0, x1), max(0.0, y1)
                        x2, y2 = min(1.0, x2), min(1.0, y2)
                        bw, bh = x2 - x1, y2 - y1
                        cx, cy = x1 + bw / 2, y1 + bh / 2
                        changed = True

                if bw * args.imgsz < 2 or bh * args.imgsz < 2:
                    rep.add("tiny_box", "{} -> {:.1f}x{:.1f}px at imgsz={}".format(
                        where, bw * args.imgsz, bh * args.imgsz, args.imgsz))
                if bw > 0.99 and bh > 0.99:
                    rep.add("full_frame_box", where)

                if not bad:
                    class_counts[cid] += 1
                    n_boxes_split += 1
                new_lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(cid, cx, cy, bw, bh))

            if changed:
                lpath.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                rep.add("clipped_and_rewritten", str(lpath.name))

        # labels with no matching image
        for lp in lbls:
            if lp.stem not in img_stems:
                rep.add("missing_image", "{}/{}".format(split, lp.name))

        per_split[split] = {"images": len(img_stems), "labels": len(lbls), "boxes": n_boxes_split}
        total_boxes += n_boxes_split

    for stem, splits in stem_to_splits.items():
        if len(set(splits)) > 1:
            rep.add("LEAKAGE_same_stem_multiple_splits", "{} in {}".format(stem, sorted(set(splits))))

    # ------------------------------ report ------------------------------ #
    print("=" * 64)
    print("DATASET VALIDATION REPORT")
    print("=" * 64)
    print("{:<8} {:>10} {:>10} {:>10}".format("SPLIT", "IMAGES", "LABELS", "OBJECTS"))
    for s in SPLITS:
        if s in per_split:
            d = per_split[s]
            print("{:<8} {:>10,} {:>10,} {:>10,}".format(s, d["images"], d["labels"], d["boxes"]))
    print("{:<8} {:>10,} {:>10,} {:>10,}".format(
        "TOTAL",
        sum(d["images"] for d in per_split.values()),
        sum(d["labels"] for d in per_split.values()),
        total_boxes))
    print()

    fatal_keys = [
        "missing_label", "missing_image", "corrupted_image", "invalid_class_id",
        "malformed_line", "non_numeric", "nonpositive_wh", "negative_value",
        "coord_out_of_range", "box_outside_image", "degenerate_image",
        "duplicate_stem_in_split", "unsupported_format", "missing_split_folder",
        "LEAKAGE_same_stem_multiple_splits",
    ]
    warn_keys = ["empty_label", "tiny_box", "full_frame_box", "clipped_and_rewritten"]

    print("ERRORS")
    print("-" * 64)
    n_fatal = 0
    for k in fatal_keys:
        c = rep.n(k)
        n_fatal += c
        print("  {:<38} {:>8,}".format(k, c))
    print()
    print("WARNINGS")
    print("-" * 64)
    for k in warn_keys:
        print("  {:<38} {:>8,}".format(k, rep.n(k)))
    print()

    shown = False
    for k in fatal_keys + warn_keys:
        if rep.details.get(k):
            if not shown:
                print("EXAMPLES")
                print("-" * 64)
                shown = True
            print("  [{}]".format(k))
            for d in rep.details[k]:
                print("      {}".format(d))
    if shown:
        print()

    if class_counts:
        print("CLASS DISTRIBUTION (valid boxes only)")
        print("-" * 64)
        mx = max(class_counts.values())
        for cid in sorted(names):
            c = class_counts.get(cid, 0)
            bar = "#" * int(28 * c / mx) if mx else ""
            flag = "   <-- NO INSTANCES" if c == 0 else ""
            print("  {:>3} {:<20} {:>7,}  {}{}".format(cid, names[cid], c, bar, flag))
        nonzero = [c for c in class_counts.values() if c > 0]
        if nonzero:
            ratio = max(nonzero) / min(nonzero)
            print("\n  Imbalance ratio (max/min): {:.1f}x".format(ratio))
            if ratio > 20:
                print("  WARNING: severe class imbalance; rare classes will score near zero mAP.")
        print()

    print("=" * 64)
    if n_fatal:
        print("Status: DATASET REQUIRES FIXING  ({:,} errors)".format(n_fatal))
        print("=" * 64)
        return 1
    print("Status: DATASET OK - ready for training")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
