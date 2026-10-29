#!/usr/bin/env python3
"""
analyze_dataset.py
==================
Step 1-3 of the thesis pipeline: inspect the raw `image/` folder and search the
whole project for annotation files. Produces a factual dataset analysis report.

This script NEVER creates, guesses or fabricates annotations. It only measures.

Usage:
    python scripts/analyze_dataset.py
    python scripts/analyze_dataset.py --images image --no-dupes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None            # allow very large images, we validate size ourselves
ImageFile.LOAD_TRUNCATED_IMAGES = False  # we WANT truncated files to raise, so we can report them

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
ANNOTATION_EXTS = {".txt", ".xml", ".json", ".csv", ".yaml", ".yml"}
ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".7z", ".rar"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Per-image worker
# --------------------------------------------------------------------------- #
def _dhash(gray: np.ndarray) -> int:
    """64-bit difference hash of a 9x8 grayscale thumbnail (near-duplicate signal)."""
    diff = gray[:, 1:] > gray[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits


def inspect_image(path_str: str) -> dict:
    """Fully decode one image and return every measurement we need. Never raises."""
    path = Path(path_str)
    rec: dict = {
        "path": path_str,
        "name": path.name,
        "ext": path.suffix.lower(),
        "ok": False,
        "error": None,
    }
    try:
        rec["bytes"] = path.stat().st_size
    except OSError as exc:
        rec["error"] = "stat failed: {}".format(exc)
        return rec

    if rec["bytes"] == 0:
        rec["error"] = "zero-byte file"
        return rec

    try:
        with Image.open(path) as im:
            rec["format"] = im.format
            rec["width"], rec["height"] = im.size
            rec["mode"] = im.mode
            try:
                rec["has_exif"] = bool(im.getexif())
            except Exception:
                rec["has_exif"] = False
            # force full decode -> catches truncation / corruption
            im.load()
            small = im.convert("L").resize((9, 8), Image.BILINEAR)
            gray9 = np.asarray(small, dtype=np.int16)
            thumb = np.asarray(im.convert("L").resize((128, 128), Image.BILINEAR), dtype=np.float32)

        rec["dhash"] = _dhash(gray9)
        rec["brightness"] = float(thumb.mean())
        # Laplacian variance on the 128x128 thumbnail -> blur / sharpness proxy
        lap = (
            -4.0 * thumb[1:-1, 1:-1]
            + thumb[:-2, 1:-1] + thumb[2:, 1:-1]
            + thumb[1:-1, :-2] + thumb[1:-1, 2:]
        )
        rec["sharpness"] = float(lap.var())

        with open(path, "rb") as fh:
            rec["md5"] = hashlib.md5(fh.read()).hexdigest()

        rec["ok"] = True
    except Exception as exc:  # noqa: BLE001 - we report whatever Pillow throws
        rec["error"] = "{}: {}".format(type(exc).__name__, exc)
    return rec


# --------------------------------------------------------------------------- #
# Near-duplicate clustering over 64-bit dhashes
# --------------------------------------------------------------------------- #
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def near_duplicate_pairs(hashes: np.ndarray, threshold: int = 5, chunk: int = 128) -> int:
    """Count pairs (i<j) whose dhash Hamming distance <= threshold."""
    n = len(hashes)
    if n < 2:
        return 0
    total = 0
    for start in range(0, n, chunk):
        block = hashes[start:start + chunk]
        xor = np.bitwise_xor(block[:, None], hashes[None, :])          # (b, n) uint64
        dist = _POPCOUNT[xor.view(np.uint8)].reshape(xor.shape + (8,)).sum(axis=2)
        rows = np.arange(start, start + len(block))[:, None]
        cols = np.arange(n)[None, :]
        total += int(((dist <= threshold) & (cols > rows)).sum())
    return total


# --------------------------------------------------------------------------- #
# Annotation discovery
# --------------------------------------------------------------------------- #
def find_annotations(root: Path) -> dict:
    hits = defaultdict(list)
    archives = []
    skip = {"venv", ".venv", "__pycache__", ".git", "node_modules", "reports", "runs"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            full = Path(dirpath) / fn
            if ext in ANNOTATION_EXTS:
                hits[ext].append(str(full.relative_to(root)))
            elif ext in ARCHIVE_EXTS:
                archives.append(str(full.relative_to(root)))
    return {"by_ext": dict(hits), "archives": archives}


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def classify_annotation_format(files: dict) -> list:
    """Best-effort identification of any annotation files that were found."""
    notes = []
    for ext, paths in files.get("by_ext", {}).items():
        for p in paths[:5]:
            full = PROJECT_ROOT / p
            try:
                head = full.read_text(errors="ignore")[:2000]
            except Exception:
                continue
            if ext == ".txt":
                lines = head.strip().splitlines()
                first = lines[0] if lines else ""
                parts = first.split()
                if len(parts) == 5 and all(_is_float(x) for x in parts):
                    notes.append("{}: looks like YOLO (class cx cy w h)".format(p))
            elif ext == ".xml" and "<annotation>" in head:
                notes.append("{}: looks like Pascal VOC XML".format(p))
            elif ext == ".json":
                if '"annotations"' in head and '"categories"' in head:
                    notes.append("{}: looks like COCO JSON".format(p))
                elif '"shapes"' in head and '"imagePath"' in head:
                    notes.append("{}: looks like LabelMe JSON".format(p))
            elif ext == ".csv":
                notes.append("{}: CSV - inspect header for xmin/ymin/xmax/ymax columns".format(p))
    return notes


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def human(n) -> str:
    return "{:,}".format(n)


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze the raw image dataset.")
    ap.add_argument("--images", default="image", help="folder containing the raw images")
    ap.add_argument("--root", default=str(PROJECT_ROOT), help="project root to scan for annotations")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--no-dupes", action="store_true", help="skip near-duplicate clustering")
    ap.add_argument("--near-dup-threshold", type=int, default=5)
    ap.add_argument("--out", default="reports", help="where to write the report files")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    img_dir = (root / args.images) if not Path(args.images).is_absolute() else Path(args.images)
    if not img_dir.is_dir():
        print("ERROR: image folder not found: {}".format(img_dir), file=sys.stderr)
        return 2

    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning {} ...".format(img_dir))
    files = sorted(p for p in img_dir.rglob("*") if p.is_file())
    image_files = [p for p in files if p.suffix.lower() in IMAGE_EXTS]
    other_files = [p for p in files if p.suffix.lower() not in IMAGE_EXTS]
    print("Found {} image files ({} non-image files) - decoding with {} workers ...".format(
        human(len(image_files)), human(len(other_files)), args.workers))

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(inspect_image, [str(p) for p in image_files], chunksize=64), 1):
            records.append(rec)
            if i % 2000 == 0:
                print("  ... {}/{}".format(human(i), human(len(image_files))))

    good = [r for r in records if r["ok"]]
    bad = [r for r in records if not r["ok"]]

    widths = np.array([r["width"] for r in good], dtype=np.int64)
    heights = np.array([r["height"] for r in good], dtype=np.int64)
    areas = widths * heights
    sizes = np.array([r["bytes"] for r in good], dtype=np.int64)
    bright = np.array([r["brightness"] for r in good], dtype=np.float64)
    sharp = np.array([r["sharpness"] for r in good], dtype=np.float64)

    ext_counts = Counter(r["ext"] for r in records)
    fmt_counts = Counter(r.get("format") for r in good)
    mode_counts = Counter(r.get("mode") for r in good)

    # exact duplicates by file content
    md5_groups = defaultdict(list)
    for r in good:
        md5_groups[r["md5"]].append(r["name"])
    exact_dup_groups = {k: v for k, v in md5_groups.items() if len(v) > 1}
    exact_dup_extra = sum(len(v) - 1 for v in exact_dup_groups.values())

    # near duplicates by dhash
    near_pairs = None
    dhash_exact_extra = None
    if not args.no_dupes and good:
        dh = np.array([r["dhash"] for r in good], dtype=np.uint64)
        dh_groups = Counter(dh.tolist())
        dhash_exact_extra = sum(c - 1 for c in dh_groups.values() if c > 1)
        print("Clustering near-duplicates (perceptual hash) ...")
        near_pairs = near_duplicate_pairs(dh, threshold=args.near_dup_threshold)

    # filename-based provenance hints
    roboflow_like = sum(1 for r in records if ".rf." in r["name"])
    name_collisions = Counter(Path(r["name"]).stem for r in records)
    stem_dupes = sum(c - 1 for c in name_collisions.values() if c > 1)

    ann = find_annotations(root)
    ann_notes = classify_annotation_format(ann)
    ann_total = sum(len(v) for v in ann["by_ext"].values())

    # ---------------- text report ---------------- #
    L = []
    A = L.append
    A("=" * 72)
    A("DATASET ANALYSIS REPORT")
    A("Object Detection and Navigation Assistance for Visually Impaired Individuals")
    A("=" * 72)
    A("Project root : {}".format(root))
    A("Image folder : {}".format(img_dir))
    A("")
    A("-" * 72)
    A("1. FILE INVENTORY")
    A("-" * 72)
    A("Total files in image folder : {}".format(human(len(files))))
    A("Recognised image files      : {}".format(human(len(image_files))))
    A("Non-image files             : {}".format(human(len(other_files))))
    for ext, c in ext_counts.most_common():
        A("    {:<12} {:>10}".format(ext or "(none)", human(c)))
    A("")
    A("Actual decoded formats (Pillow):")
    for fmt, c in fmt_counts.most_common():
        A("    {:<12} {:>10}".format(str(fmt), human(c)))
    A("")
    A("Colour modes:")
    for m, c in mode_counts.most_common():
        A("    {:<12} {:>10}".format(str(m), human(c)))
    A("")
    A("-" * 72)
    A("2. RESOLUTION STATISTICS  (readable images only)")
    A("-" * 72)
    if len(good):
        i_min, i_max = int(areas.argmin()), int(areas.argmax())
        A("Readable images         : {}".format(human(len(good))))
        A("Minimum resolution      : {} x {}  ({} px)  [{}]".format(
            widths[i_min], heights[i_min], human(int(areas[i_min])), good[i_min]["name"]))
        A("Maximum resolution      : {} x {}  ({} px)  [{}]".format(
            widths[i_max], heights[i_max], human(int(areas[i_max])), good[i_max]["name"]))
        A("Average resolution      : {:.1f} x {:.1f}".format(widths.mean(), heights.mean()))
        A("Median  resolution      : {:.0f} x {:.0f}".format(np.median(widths), np.median(heights)))
        A("Width  min/med/max      : {} / {:.0f} / {}".format(widths.min(), np.median(widths), widths.max()))
        A("Height min/med/max      : {} / {:.0f} / {}".format(heights.min(), np.median(heights), heights.max()))
        ar = widths / np.maximum(heights, 1)
        A("Aspect ratio min/med/max: {:.3f} / {:.3f} / {:.3f}".format(ar.min(), np.median(ar), ar.max()))
        A("")
        A("Most common exact resolutions:")
        for (w, h), c in Counter(zip(widths.tolist(), heights.tolist())).most_common(10):
            A("    {:>5} x {:<5}  {:>8}  ({:5.2f}%)".format(w, h, human(c), 100.0 * c / len(good)))
        A("")
        A("Size buckets (longest side):")
        maxside = np.maximum(widths, heights)
        buckets = [(0, 128), (128, 256), (256, 416), (416, 640), (640, 1024), (1024, 1920), (1920, 10 ** 9)]
        for lo, hi in buckets:
            c = int(((maxside > lo) & (maxside <= hi)).sum())
            label = ">{}".format(lo) if hi >= 10 ** 9 else "{}-{}".format(lo + 1, hi)
            A("    {:<12} {:>8}  ({:5.2f}%)".format(label, human(c), 100.0 * c / len(good)))
        A("")
        A("File size min/avg/max   : {:.1f} KB / {:.1f} KB / {:.2f} MB".format(
            sizes.min() / 1024, sizes.mean() / 1024, sizes.max() / 1024 / 1024))
        A("Total dataset size      : {:.2f} GB".format(sizes.sum() / 1024 / 1024 / 1024))
    else:
        A("No readable images.")
    A("")
    A("-" * 72)
    A("3. IMAGE QUALITY INDICATORS")
    A("-" * 72)
    if len(good):
        A("Mean brightness (0-255) min/avg/max : {:.1f} / {:.1f} / {:.1f}".format(
            bright.min(), bright.mean(), bright.max()))
        dark = int((bright < 60).sum())
        brightc = int((bright > 200).sum())
        A("Very dark images   (mean < 60)     : {}  ({:.2f}%)".format(human(dark), 100.0 * dark / len(good)))
        A("Very bright images (mean > 200)    : {}  ({:.2f}%)".format(human(brightc), 100.0 * brightc / len(good)))
        blurry = int((sharp < 50).sum())
        A("Laplacian variance median          : {:.1f}".format(np.median(sharp)))
        A("Likely blurry / low-detail (<50)   : {}  ({:.2f}%)".format(human(blurry), 100.0 * blurry / len(good)))
        gray = sum(c for m, c in mode_counts.items() if m in {"L", "1", "LA"})
        A("Grayscale images                   : {}".format(human(gray)))
        exif = sum(1 for r in good if r.get("has_exif"))
        A("Images carrying EXIF metadata      : {}".format(human(exif)))
    A("")
    A("-" * 72)
    A("4. CORRUPTED / UNREADABLE IMAGES")
    A("-" * 72)
    A("Corrupted or unreadable : {}".format(human(len(bad))))
    for r in bad[:25]:
        A("    {}  ->  {}".format(r["name"], r["error"]))
    if len(bad) > 25:
        A("    ... and {} more (see JSON report)".format(human(len(bad) - 25)))
    A("")
    A("-" * 72)
    A("5. DUPLICATE ANALYSIS  (data-leakage risk)")
    A("-" * 72)
    A("Duplicate filename stems (same stem, different extension) : {}".format(human(stem_dupes)))
    A("Byte-identical duplicate groups (MD5)                     : {}".format(human(len(exact_dup_groups))))
    A("Redundant copies from byte-identical duplicates           : {}".format(human(exact_dup_extra)))
    if dhash_exact_extra is not None:
        A("Visually identical copies (identical perceptual hash)     : {}".format(human(dhash_exact_extra)))
    if near_pairs is not None:
        A("Near-duplicate image PAIRS (dhash Hamming <= {})           : {}".format(
            args.near_dup_threshold, human(near_pairs)))
    if exact_dup_groups:
        A("")
        A("Examples of byte-identical groups:")
        for k, v in list(exact_dup_groups.items())[:5]:
            A("    {}...: {}{}".format(k[:12], v[:4], " ..." if len(v) > 4 else ""))
    A("")
    A("-" * 72)
    A("6. ANNOTATION SEARCH  (entire project tree)")
    A("-" * 72)
    A("Searched for: .txt .xml .json .csv .yaml .yml  under {}".format(root))
    if ann_total == 0:
        A("Annotation files found : 0")
        A("Archives found         : {}".format(", ".join(ann["archives"]) if ann["archives"] else "0"))
        A("")
        A("  >>> NO ANNOTATIONS OF ANY KIND EXIST IN THIS PROJECT. <<<")
        A("  No YOLO .txt, no COCO JSON, no Pascal VOC XML, no LabelMe, no CSV.")
    else:
        for ext, paths in ann["by_ext"].items():
            A("    {}: {} file(s); e.g. {}".format(ext, human(len(paths)), paths[:3]))
        if ann_notes:
            A("")
            A("Format identification:")
            for n in ann_notes:
                A("    {}".format(n))
    A("")
    A("Filenames containing Roboflow's '.rf.' marker : {} ({:.1f}%)".format(
        human(roboflow_like), 100.0 * roboflow_like / max(len(records), 1)))
    A("  (A '.rf.<hash>' filename means the file was exported from Roboflow.")
    A("   The image was exported WITHOUT its label file, or labels were not copied.)")
    A("")
    A("-" * 72)
    A("7. SUITABILITY FOR OBJECT DETECTION")
    A("-" * 72)
    if ann_total == 0:
        A("Dataset status : NOT TRAINABLE YET (images only, no ground truth)")
        A("")
        A("Images alone cannot train a supervised YOLO detector. Every object you")
        A("want the model to detect must have (a) a bounding box and (b) a class id.")
        A("Required next step: bounding-box annotation. See README.md, section")
        A("'Annotation Workflow'.")
    else:
        A("Annotation files were found - identify the format and convert to YOLO.")
    A("")
    A("=" * 72)
    A("END OF REPORT - every number above was measured from the actual files.")
    A("=" * 72)

    text = "\n".join(L)
    print("\n" + text)

    (out_dir / "dataset_analysis.txt").write_text(text, encoding="utf-8")

    json_payload = {
        "root": str(root),
        "image_dir": str(img_dir),
        "total_files": len(files),
        "image_files": len(image_files),
        "readable": len(good),
        "corrupted": [{"name": r["name"], "error": r["error"]} for r in bad],
        "ext_counts": dict(ext_counts),
        "format_counts": {str(k): v for k, v in fmt_counts.items()},
        "mode_counts": {str(k): v for k, v in mode_counts.items()},
        "resolution": {
            "min_wh": [int(widths.min()), int(heights.min())] if len(good) else None,
            "max_wh": [int(widths.max()), int(heights.max())] if len(good) else None,
            "mean_wh": [float(widths.mean()), float(heights.mean())] if len(good) else None,
            "median_wh": [float(np.median(widths)), float(np.median(heights))] if len(good) else None,
            "top_resolutions": [
                {"w": w, "h": h, "count": c}
                for (w, h), c in Counter(zip(widths.tolist(), heights.tolist())).most_common(20)
            ] if len(good) else [],
        },
        "duplicates": {
            "byte_identical_groups": len(exact_dup_groups),
            "byte_identical_redundant": exact_dup_extra,
            "dhash_identical_redundant": dhash_exact_extra,
            "near_duplicate_pairs": near_pairs,
            "duplicate_filename_stems": stem_dupes,
        },
        "annotations": {
            "found": ann_total,
            "by_ext": {k: len(v) for k, v in ann["by_ext"].items()},
            "notes": ann_notes,
            "archives": ann["archives"],
        },
        "roboflow_named_files": roboflow_like,
        "trainable": ann_total > 0,
    }
    (out_dir / "dataset_analysis.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    # duplicate manifest, so prepare_dataset.py can exclude leakage
    if exact_dup_groups:
        with open(out_dir / "duplicate_groups.txt", "w", encoding="utf-8") as fh:
            for k, v in exact_dup_groups.items():
                fh.write(k + "\t" + "\t".join(v) + "\n")

    print("\nSaved: {}".format(out_dir / "dataset_analysis.txt"))
    print("Saved: {}".format(out_dir / "dataset_analysis.json"))
    if exact_dup_groups:
        print("Saved: {}".format(out_dir / "duplicate_groups.txt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
