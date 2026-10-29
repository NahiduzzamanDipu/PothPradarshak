#!/usr/bin/env python3
"""
build_annotation_set.py
=======================
STEP 0 of the annotation workflow.

The raw `image/` folder is a merge of ~6 unrelated source datasets (see
reports/provenance_audit.txt). Only the Dhaka footpath group is in scope for
this thesis. This script:

  1. selects that group,
  2. clusters near-duplicate frames (perceptual hash) and keeps ONE
     representative per cluster,
  3. copies the survivors to `annotation_input/` ready for upload to Roboflow,
  4. writes reports/annotation_set_manifest.csv recording every cluster so the
     decision is reproducible and defensible in the viva.

Why de-duplicate BEFORE annotating, not after:
  * you do not pay to annotate the same scene twice, and
  * a frame in train with its near-copy in test inflates mAP. Removing the
    copies up front makes leakage structurally impossible.

Usage:
    python scripts/build_annotation_set.py
    python scripts/build_annotation_set.py --threshold 8      # stricter dedup
    python scripts/build_annotation_set.py --keep-duplicates  # no dedup
    python scripts/build_annotation_set.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "image"

# The Dhaka footpath group, identified in the provenance audit:
# purely numeric .jpg filenames, exactly 256x256.
RE_DHAKA = re.compile(r"^\d+\.jpg$", re.I)
DHAKA_SIZE = (256, 256)


def probe(path_str: str):
    try:
        with Image.open(path_str) as im:
            w, h = im.size
            g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    except Exception:
        return path_str, -1, -1, None
    bits = 0
    for b in (g[:, 1:] > g[:, :-1]).flatten():
        bits = (bits << 1) | int(b)
    return path_str, w, h, bits


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


class DisjointSet:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster(hashes, threshold, chunk=256):
    n = len(hashes)
    ds = DisjointSet(n)
    arr = np.array(hashes, dtype=np.uint64)
    for s in range(0, n, chunk):
        block = arr[s:s + chunk]
        xor = np.bitwise_xor(block[:, None], arr[None, :])
        dist = _POPCOUNT[xor.view(np.uint8)].reshape(xor.shape + (8,)).sum(axis=2)
        rows, cols = np.where(dist <= threshold)
        for r, c in zip(rows, cols):
            if s + r < c:
                ds.union(int(s + r), int(c))
    return [ds.find(i) for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the de-duplicated annotation set.")
    ap.add_argument("--out", default="annotation_input")
    ap.add_argument("--threshold", type=int, default=5,
                    help="dhash Hamming distance below which two frames count as duplicates")
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not SRC.is_dir():
        print("ERROR: {} not found.".format(SRC))
        return 2

    candidates = [p for p in sorted(SRC.iterdir()) if p.is_file() and RE_DHAKA.match(p.name)]
    print("Numeric .jpg candidates : {:,}".format(len(candidates)))

    with ProcessPoolExecutor() as pool:
        probed = list(pool.map(probe, [str(p) for p in candidates], chunksize=64))

    selected = [(Path(p), b) for p, w, h, b in probed if (w, h) == DHAKA_SIZE and b is not None]
    rejected_size = Counter((w, h) for p, w, h, b in probed if (w, h) != DHAKA_SIZE)
    print("Matching {}x{}        : {:,}".format(DHAKA_SIZE[0], DHAKA_SIZE[1], len(selected)))
    if rejected_size:
        print("Excluded (wrong size)   : {:,}  e.g. {}".format(
            sum(rejected_size.values()), rejected_size.most_common(3)))
    if not selected:
        print("ERROR: no images matched the Dhaka group.")
        return 2

    paths = [p for p, _ in selected]
    hashes = [b for _, b in selected]

    if args.keep_duplicates:
        groups = list(range(len(paths)))
        print("De-duplication          : DISABLED")
    else:
        print("Clustering near-duplicates (Hamming <= {}) ...".format(args.threshold))
        groups = cluster(hashes, args.threshold)

    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)

    # keep the lexicographically first member of each cluster as representative
    keep_idx, drop_idx = [], []
    for g, members in by_group.items():
        members = sorted(members, key=lambda i: paths[i].name)
        keep_idx.append(members[0])
        drop_idx.extend(members[1:])
    keep_idx.sort()

    sizes = Counter(len(v) for v in by_group.values())
    print()
    print("=" * 60)
    print("DE-DUPLICATION RESULT")
    print("=" * 60)
    print("  input images        : {:,}".format(len(paths)))
    print("  unique clusters     : {:,}".format(len(by_group)))
    print("  kept  (to annotate) : {:,}".format(len(keep_idx)))
    print("  dropped (duplicates): {:,}  ({:.1f}%)".format(
        len(drop_idx), 100.0 * len(drop_idx) / len(paths)))
    print()
    print("  cluster size distribution:")
    for size in sorted(sizes)[:8]:
        print("    {} image(s) per cluster : {:,} clusters".format(size, sizes[size]))
    biggest = max(by_group.values(), key=len)
    if len(biggest) > 1:
        print("  largest cluster: {} frames, e.g. {}".format(
            len(biggest), [paths[i].name for i in sorted(biggest)[:5]]))
    print("=" * 60)

    (ROOT / "reports").mkdir(exist_ok=True)
    manifest = ROOT / "reports" / "annotation_set_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["filename", "cluster_id", "cluster_size", "decision"])
        keep_set = set(keep_idx)
        for i, p in enumerate(paths):
            wr.writerow([p.name, groups[i], len(by_group[groups[i]]),
                         "KEEP" if i in keep_set else "DROP_near_duplicate"])
    print("Manifest: {}".format(manifest))

    if args.dry_run:
        print("\n--dry-run: no files copied.")
        return 0

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    for i in keep_idx:
        shutil.copy2(paths[i], out / paths[i].name)

    print("\nCopied {:,} images -> {}".format(len(keep_idx), out))
    print("\nNEXT STEP")
    print("  1. Upload the contents of '{}' to a new Roboflow project".format(args.out))
    print("     (Project type: Object Detection).")
    print("  2. Annotate with the 9 classes declared in data.yaml.")
    print("  3. Export as 'YOLOv11' / 'YOLOv8' format, download the zip.")
    print("  4. Put images+labels into annotated/images and annotated/labels,")
    print("     then run: python scripts/prepare_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
