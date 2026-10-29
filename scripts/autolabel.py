#!/usr/bin/env python3
"""
autolabel.py
============
STEP 1 of the annotation workflow: generate CANDIDATE labels automatically.

    build_annotation_set.py  ->  autolabel.py  ->  review_labels.py  ->  train.py
       (choose images)          (this script)      (human verifies)

=============================================================================
WHAT THIS IS, AND WHAT IT IS NOT
=============================================================================
This produces MACHINE-GENERATED, UNVERIFIED candidate boxes using an
open-vocabulary detector (YOLO-World). They are a starting point for a human,
not ground truth.

Training directly on this output without review would mean the thesis model
learns another model's mistakes, and any mAP reported against these labels
would be measuring agreement with YOLO-World rather than correctness. Every
label written here is therefore marked `verified: false` in the manifest until
a human passes it through `review_labels.py`.

The honest description for the methodology chapter is:
    "Machine-assisted annotation: candidate bounding boxes were generated with
     an open-vocabulary detector (YOLO-World v2) and every image was then
     manually reviewed, with incorrect boxes corrected or deleted before
     training."

=============================================================================
PER-CLASS RELIABILITY - READ BEFORE TRUSTING THE OUTPUT
=============================================================================
Open-vocabulary detectors are strong on classes that resemble their training
distribution and weak on the rest. Measured on this dataset:

  RELIABLE     person, parked_vehicle, tree
               Common COCO-like objects. Expect mostly-correct boxes.

  PARTIAL      pole, stairs, street_obstacle
               Found when prominent, missed when cluttered or partial.

  UNRELIABLE   pothole, open_drain, curb
               Ground-surface defects have no clean visual prototype and are
               poorly separated from shadows, patches and ordinary paving.
               Expect to draw most of these yourself. They are also the
               HIGHEST-PRIORITY classes for the thesis, so budget review time
               here rather than on the easy classes.

`--report` prints the achieved per-class counts so this table can be replaced
with the numbers actually observed on your data.

Usage:
    python scripts/autolabel.py                       # all of annotation_input/
    python scripts/autolabel.py --limit 50            # quick trial run
    python scripts/autolabel.py --conf 0.10           # more, noisier candidates
    python scripts/autolabel.py --device cpu          # force CPU
    python scripts/autolabel.py --report              # summarise existing output
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "annotation_input"
DEFAULT_OUT = ROOT / "autolabels"
MODEL_CANDIDATES = [
    ROOT / "models" / "yolov8x-worldv2.pt",
    ROOT / "yolov8x-worldv2.pt",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --------------------------------------------------------------------------- #
# Prompt design
# --------------------------------------------------------------------------- #
# Each thesis class maps to SEVERAL natural-language prompts. Open-vocabulary
# detectors respond to descriptive phrases far better than to bare class names
# ("utility pole on footpath" >> "pole"), and several phrasings per class
# recover objects that any single phrasing misses.
#
# The prompt index -> class id mapping is what converts YOLO-World's output
# back into the thesis taxonomy, so PROMPTS and PROMPT_CLASS_ID must stay
# aligned. Both are derived from one table to make that impossible to break.
# --------------------------------------------------------------------------- #
CLASS_PROMPTS: dict[int, tuple[str, list[str]]] = {
    0: ("person", [
        "person", "pedestrian walking", "man standing", "woman walking",
    ]),
    1: ("pole", [
        "utility pole", "electric pole on footpath", "lamp post", "sign post",
    ]),
    2: ("tree", [
        "tree trunk", "tree growing on pavement",
    ]),
    3: ("stairs", [
        "stairs", "concrete steps", "staircase entrance",
    ]),
    4: ("pothole", [
        "pothole", "broken pavement", "hole in the road surface",
        "damaged footpath tiles",
    ]),
    5: ("open_drain", [
        "open drain", "uncovered drain", "manhole without cover",
        "open sewer gap in pavement",
    ]),
    6: ("parked_vehicle", [
        "parked car", "rickshaw", "auto rickshaw", "parked motorcycle",
        "parked van", "bicycle parked on footpath",
    ]),
    7: ("street_obstacle", [
        "street vendor stall", "pile of bricks", "construction material",
        "garbage bin", "sand pile on footpath", "signboard on pavement",
    ]),
    8: ("curb", [
        "kerb edge of footpath", "raised pavement edge",
    ]),
}

# Classes where the detector is known to be weak. Flagged per image so the
# review tool can route these to a human first.
LOW_TRUST_CLASSES = {4, 5, 8}


# --------------------------------------------------------------------------- #
# Candidate filtering
# --------------------------------------------------------------------------- #
# Open-vocabulary detectors fail in two mechanical ways on this data, both
# visible when the raw output is plotted:
#
#   1. WHOLE-IMAGE BOXES. Asked for "tree" or "street vendor stall", the model
#      sometimes returns a box covering the entire frame - it has matched the
#      scene, not an object. Useless as a label and slow for a human to delete
#      one at a time.
#   2. STACKED DUPLICATES. Several prompts map to one class (e.g. "parked car"
#      and "rickshaw"), so the same vehicle is returned two or three times.
#      Ultralytics' NMS runs per PROMPT, so it never sees these as duplicates.
#      Once prompts are collapsed into classes, they must be suppressed again.
#
# Both are removed here rather than left for the reviewer. A box that survives
# is still only a candidate - this filters obvious machine errors, it does not
# make anything verified.

MAX_AREA_RATIO = 0.55     # a box larger than this is describing the scene
MIN_AREA_RATIO = 0.0004   # ~13x13 px in a 640 frame: too far away to matter
NMS_IOU = 0.65            # same-class overlap treated as a duplicate


def _iou(a: dict, b: dict) -> float:
    ax1, ay1 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2
    ax2, ay2 = a["cx"] + a["w"] / 2, a["cy"] + a["h"] / 2
    bx1, by1 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2
    bx2, by2 = b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2
    ix, iy = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def filter_candidates(boxes: list[dict], max_area: float = MAX_AREA_RATIO) -> tuple[list[dict], Counter]:
    """Drop scene-sized and speck-sized boxes, then de-duplicate per class."""
    dropped: Counter = Counter()

    kept = []
    for b in boxes:
        area = b["w"] * b["h"]
        if area > max_area:
            dropped["too_large"] += 1
            continue
        if area < MIN_AREA_RATIO:
            dropped["too_small"] += 1
            continue
        kept.append(b)

    # Greedy NMS within each class, highest confidence first.
    kept.sort(key=lambda b: b["conf"], reverse=True)
    final: list[dict] = []
    for b in kept:
        if any(o["cls"] == b["cls"] and _iou(o, b) > NMS_IOU for o in final):
            dropped["duplicate"] += 1
            continue
        final.append(b)
    return final, dropped


def build_prompt_tables() -> tuple[list[str], list[int], dict[int, str]]:
    prompts: list[str] = []
    prompt_class_id: list[int] = []
    names: dict[int, str] = {}
    for cid, (name, phrases) in sorted(CLASS_PROMPTS.items()):
        names[cid] = name
        for phrase in phrases:
            prompts.append(phrase)
            prompt_class_id.append(cid)
    return prompts, prompt_class_id, names


def find_model(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return p
        raise SystemExit("Model not found: {}".format(p))
    for c in MODEL_CANDIDATES:
        if c.exists():
            return c
    raise SystemExit(
        "YOLO-World weights not found. Expected one of:\n  " +
        "\n  ".join(str(c) for c in MODEL_CANDIDATES) +
        "\nDownload with:\n"
        "  python -c \"from ultralytics import YOLO; YOLO('yolov8x-worldv2.pt')\"\n"
        "then move the file into models/."
    )


def resolve_device(pref: str) -> str:
    if pref == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if pref == "cuda":
            print("[warn] --device cuda requested but CUDA is unavailable; using CPU.")
    except Exception as exc:
        print("[warn] could not query torch: {}".format(exc))
    return "cpu"


def iter_images(src: Path):
    return sorted(p for p in src.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


# --------------------------------------------------------------------------- #
def run(args) -> int:
    src = Path(args.source) if args.source else DEFAULT_SRC
    out = Path(args.output) if args.output else DEFAULT_OUT
    if not src.is_dir():
        raise SystemExit(
            "Source folder not found: {}\nRun scripts/build_annotation_set.py first."
            .format(src))

    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.csv"

    images = iter_images(src)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit("No images found in {}".format(src))

    # Resume: skip images that already have a label file, unless overwriting.
    done = set()
    if manifest_path.exists() and not args.overwrite:
        with manifest_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done.add(row["image"])
    todo = [p for p in images if p.name not in done]

    print("=" * 70)
    print("AUTO-LABELLING (machine-generated candidates - NOT ground truth)")
    print("=" * 70)
    print("  source     : {}".format(src))
    print("  output     : {}".format(out))
    print("  images     : {} total, {} already done, {} to process"
          .format(len(images), len(images) - len(todo), len(todo)))
    if not todo:
        print("\nNothing to do. Use --overwrite to redo, or --report to summarise.")
        return 0

    from ultralytics import YOLO

    model_path = find_model(args.model)
    device = resolve_device(args.device)
    prompts, prompt_class_id, names = build_prompt_tables()

    print("  model      : {}".format(model_path.name))
    print("  device     : {}".format("GPU" if device != "cpu" else "CPU"))
    print("  prompts    : {} phrases -> {} classes".format(len(prompts), len(names)))
    print("  conf       : {}".format(args.conf))
    print()

    model = YOLO(str(model_path))
    model.set_classes(prompts)

    write_header = not manifest_path.exists() or args.overwrite
    mode = "w" if write_header else "a"
    fh = manifest_path.open(mode, newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if write_header:
        writer.writerow(["image", "n_boxes", "classes", "max_conf",
                         "needs_attention", "verified"])

    class_counts: Counter = Counter()
    drop_counts: Counter = Counter()
    t_start = time.time()
    images_with_boxes = 0

    for i, img_path in enumerate(todo, 1):
        try:
            results = model.predict(
                source=str(img_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                max_det=args.max_det,
                verbose=False,
            )
        except Exception as exc:
            print("[error] {}: {}".format(img_path.name, exc))
            writer.writerow([img_path.name, 0, "", 0.0, 1, 0])
            continue

        res = results[0]
        boxes = getattr(res, "boxes", None)
        lines: list[str] = []
        found: Counter = Counter()
        max_conf = 0.0
        raw: list[dict] = []

        if boxes is not None and len(boxes):
            xywhn = boxes.xywhn.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            prompt_ids = boxes.cls.cpu().numpy().astype(int)

            for (cx, cy, w, h), cf, pid in zip(xywhn, confs, prompt_ids):
                if pid < 0 or pid >= len(prompt_class_id):
                    continue
                cid = prompt_class_id[pid]      # prompt -> thesis class
                # Clamp: YOLO format requires everything inside [0,1].
                cx, cy = min(max(float(cx), 0.0), 1.0), min(max(float(cy), 0.0), 1.0)
                w, h = min(max(float(w), 0.0), 1.0), min(max(float(h), 0.0), 1.0)
                if w <= 0.001 or h <= 0.001:
                    continue
                raw.append({"cls": cid, "cx": cx, "cy": cy, "w": w, "h": h,
                            "conf": float(cf)})

        # Remove scene-sized boxes and cross-prompt duplicates before a human
        # ever sees them (see filter_candidates).
        filtered, dropped = filter_candidates(raw, max_area=args.max_area)
        for k, v in dropped.items():
            drop_counts[k] += v

        for b in filtered:
            lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                b["cls"], b["cx"], b["cy"], b["w"], b["h"]))
            found[b["cls"]] += 1
            class_counts[b["cls"]] += 1
            max_conf = max(max_conf, b["conf"])

        # An empty .txt is a valid YOLO label meaning "no objects". It is
        # written deliberately so the reviewer sees the image was processed
        # rather than skipped.
        (labels_dir / (img_path.stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if lines:
            images_with_boxes += 1

        # Flag images a human should look at first: nothing found at all, or
        # a class the detector is known to be bad at.
        needs_attention = int(
            (not lines) or any(c in LOW_TRUST_CLASSES for c in found))

        writer.writerow([
            img_path.name, len(lines),
            ";".join("{}:{}".format(names[c], n) for c, n in sorted(found.items())),
            round(max_conf, 4), needs_attention, 0,
        ])

        if i % 25 == 0 or i == len(todo):
            fh.flush()
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed else 0
            eta = (len(todo) - i) / rate if rate else 0
            print("  [{:>5}/{}] {:.2f} img/s  ETA {:>5.1f} min  boxes so far: {}"
                  .format(i, len(todo), rate, eta / 60, sum(class_counts.values())))

    fh.close()

    elapsed = time.time() - t_start
    print()
    print("=" * 70)
    print("DONE in {:.1f} min".format(elapsed / 60))
    print("  images processed     : {}".format(len(todo)))
    print("  images with >=1 box  : {} ({:.0%})".format(
        images_with_boxes, images_with_boxes / max(len(todo), 1)))
    print("  candidate boxes      : {}".format(sum(class_counts.values())))
    if drop_counts:
        print("  filtered out         : {} ({})".format(
            sum(drop_counts.values()),
            ", ".join("{} {}".format(v, k) for k, v in drop_counts.most_common())))
    print()
    print("  per class (CANDIDATES, unverified):")
    for cid in sorted(names):
        n = class_counts.get(cid, 0)
        flag = "  <- low-trust, expect to fix these by hand" if cid in LOW_TRUST_CLASSES else ""
        print("    {:>2} {:<16} {:>6}{}".format(cid, names[cid], n, flag))
    print("=" * 70)

    (out / "autolabel_meta.json").write_text(json.dumps({
        "model": model_path.name,
        "device": device,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "prompts": {names[c]: CLASS_PROMPTS[c][1] for c in sorted(names)},
        "images_processed": len(todo),
        "candidate_boxes": sum(class_counts.values()),
        "filtered_out": dict(drop_counts),
        "max_area_ratio": args.max_area,
        "per_class": {names[c]: class_counts.get(c, 0) for c in sorted(names)},
        "verified": False,
        "note": "Machine-generated candidates from an open-vocabulary detector. "
                "NOT ground truth. Must pass through scripts/review_labels.py.",
    }, indent=2), encoding="utf-8")

    print("\nNEXT STEP")
    print("  python scripts/review_labels.py      # verify and correct, then export")
    return 0


# --------------------------------------------------------------------------- #
def report(args) -> int:
    out = Path(args.output) if args.output else DEFAULT_OUT
    manifest = out / "manifest.csv"
    if not manifest.exists():
        raise SystemExit("No manifest at {}. Run autolabel first.".format(manifest))

    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    counts: Counter = Counter()
    verified = 0
    attention = 0
    empty = 0
    for r in rows:
        if r.get("verified") == "1":
            verified += 1
        if r.get("needs_attention") == "1":
            attention += 1
        if int(r["n_boxes"]) == 0:
            empty += 1
        for part in (r["classes"] or "").split(";"):
            if ":" in part:
                name, n = part.rsplit(":", 1)
                counts[name] += int(n)

    print("=" * 70)
    print("AUTO-LABEL REPORT   {}".format(manifest))
    print("=" * 70)
    print("  images            : {}".format(len(rows)))
    print("  with no boxes     : {} ({:.0%})".format(empty, empty / max(len(rows), 1)))
    print("  needing attention : {}".format(attention))
    print("  human-verified    : {} / {}".format(verified, len(rows)))
    print("  candidate boxes   : {}".format(sum(counts.values())))
    print()
    for name, n in counts.most_common():
        print("    {:<18} {:>6}".format(name, n))
    print("=" * 70)
    if verified < len(rows):
        print("\n{} images are still UNVERIFIED. Do not train on these yet:"
              .format(len(rows) - verified))
        print("  python scripts/review_labels.py")
    return 0


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate candidate YOLO labels with an open-vocabulary detector.")
    ap.add_argument("--source", default=None, help="folder of images (default: annotation_input/)")
    ap.add_argument("--output", default=None, help="output folder (default: autolabels/)")
    ap.add_argument("--model", default=None, help="path to YOLO-World weights")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--conf", type=float, default=0.08,
                    help="low on purpose: a human deletes false boxes far faster "
                         "than they draw missing ones (default: 0.08)")
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-det", type=int, default=40)
    ap.add_argument("--max-area", type=float, default=MAX_AREA_RATIO,
                    help="drop boxes covering more than this fraction of the "
                         "frame; they describe the scene, not an object "
                         "(default: {})".format(MAX_AREA_RATIO))
    ap.add_argument("--limit", type=int, default=0, help="process only the first N images")
    ap.add_argument("--overwrite", action="store_true", help="redo images already done")
    ap.add_argument("--report", action="store_true", help="summarise existing output and exit")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.report:
        return report(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
