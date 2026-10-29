#!/usr/bin/env python3
"""
validate.py
===========
Evaluate the trained model and print thesis metrics:
Precision, Recall, mAP@50, mAP@50-95 and a confusion matrix.

    python scripts/validate.py

This is the zero-argument entry point. It finds the trained weights the same
way the application does, so there is no path to remember and no way to
accidentally evaluate the wrong run. The heavy lifting is done by
`evaluate.py`, which is still available directly when you want control:

    python scripts/evaluate.py --weights runs/train/exp/weights/best.pt --split val

Anything passed here is forwarded on:

    python scripts/validate.py --split val
    python scripts/validate.py --benchmark
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Same search order as backend/config.py - first hit wins. Kept in step with
# the application so "the model validate.py scored" and "the model the app
# runs" can never diverge.
WEIGHT_GLOBS = [
    "models/best.pt",
    "runs/train/*/weights/best.pt",
    "runs/detect/*/weights/best.pt",
    "runs/*/weights/best.pt",
    "runs/**/weights/best.pt",
    "best.pt",
]


def discover_weights() -> Path | None:
    """Newest matching best.pt, so a re-run is evaluated rather than the first."""
    hits: list[Path] = []
    for pattern in WEIGHT_GLOBS:
        hits.extend(p for p in ROOT.glob(pattern) if p.is_file())
    if not hits:
        return None
    # Deduplicate while keeping newest-first ordering.
    unique = {p.resolve(): p for p in hits}
    return max(unique.values(), key=lambda p: p.stat().st_mtime)


def dataset_ready() -> bool:
    """True when prepare_dataset.py has actually produced the split folders."""
    import yaml
    cfg_path = ROOT / "data.yaml"
    if not cfg_path.exists():
        return False
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return False
    base = (cfg_path.parent / cfg.get("path", ".")).resolve()
    for split in ("val", "test"):
        rel = cfg.get(split)
        if not rel or not (base / rel).is_dir():
            return False
    return True


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Respect an explicit --weights if the user supplied one.
    if not any(a == "--weights" or a.startswith("--weights=") for a in argv):
        weights = discover_weights()
        if weights is None:
            print("=" * 70)
            print("NO TRAINED MODEL FOUND")
            print("=" * 70)
            print("Searched, relative to {}:".format(ROOT))
            for g in WEIGHT_GLOBS:
                print("    {}".format(g))
            print()
            print("There is nothing to validate until training has produced a best.pt.")
            print("The COCO placeholder the app falls back to is deliberately NOT")
            print("evaluated here: scoring it would produce numbers for a model that")
            print("has never seen the thesis classes, which is worse than no numbers.")
            print()
            print("Finish the pipeline first:")
            print("    python scripts/review_labels.py     # verify labels")
            print("    python scripts/prepare_dataset.py   # build the splits")
            print("    python scripts/train.py             # produce best.pt")
            return 1
        print("Using auto-discovered weights: {}".format(
            weights.relative_to(ROOT) if weights.is_relative_to(ROOT) else weights))
        argv = ["--weights", str(weights)] + argv

    # Fail with an explanation rather than a 30-line Ultralytics traceback when
    # the splits have not been built yet. Same spirit as train.py's preflight.
    if not dataset_ready():
        print("=" * 70)
        print("NO DATASET TO EVALUATE AGAINST")
        print("=" * 70)
        print("Found trained weights, but dataset/images/{train,val,test} do not")
        print("exist, so there is nothing to score the model on.")
        print()
        print("    python scripts/prepare_dataset.py    # build the splits")
        return 1

    # evaluate.main() parses sys.argv directly and takes no parameters, so the
    # forwarded arguments are installed there rather than passed in.
    import evaluate
    saved = sys.argv
    sys.argv = [str(Path(evaluate.__file__).name)] + argv
    try:
        return evaluate.main()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    sys.exit(main())
