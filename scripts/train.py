#!/usr/bin/env python3
"""
train.py
========
Train a YOLO detector on the prepared dataset.

This version includes controlled data augmentation for improving
generalization on varied outdoor/footpath images.

All training parameters are saved to:
runs/train/<name>/hyperparameters.json

The script refuses to start if the dataset is missing or unlabelled.

Usage:
    python scripts/train.py

    python scripts/train.py --model yolo11n.pt --epochs 100 --batch 8

    python scripts/train.py --imgsz 640 --name augmented_640

    python scripts/train.py --imgsz 768 --name augmented_768

    python scripts/train.py --model path/to/best.pt --name augmented_model
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import resolve_for_ultralytics  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Default training configuration
# ---------------------------------------------------------------------------

MODEL = "yolo11n.pt"

IMG_SIZE = 640
EPOCHS = 100
BATCH_SIZE = 16
PATIENCE = 30

LEARNING_RATE = 0.01
SEED = 42


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

def build_args() -> argparse.Namespace:

    ap = argparse.ArgumentParser(
        description="Train a YOLO object detector with data augmentation."
    )

    # Dataset / model
    ap.add_argument(
        "--data",
        default="data.yaml"
    )

    ap.add_argument(
        "--model",
        default=MODEL
    )

    # Training
    ap.add_argument(
        "--imgsz",
        type=int,
        default=IMG_SIZE
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=BATCH_SIZE
    )

    ap.add_argument(
        "--patience",
        type=int,
        default=PATIENCE
    )

    # Learning rate
    ap.add_argument(
        "--lr0",
        type=float,
        default=LEARNING_RATE
    )

    ap.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate = lr0 * lrf"
    )

    # Optimizer
    ap.add_argument(
        "--optimizer",
        default="auto",
        choices=["auto", "SGD", "Adam", "AdamW"]
    )

    # Hardware
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Dataloader workers; default is 0 on Windows and 8 elsewhere"
    )

    ap.add_argument(
        "--device",
        default=None,
        help="'cuda', 'cpu', '0', '0,1'; default = auto"
    )

    # Output
    ap.add_argument(
        "--project",
        default="runs/train"
    )

    ap.add_argument(
        "--name",
        default="exp"
    )

    # Reproducibility
    ap.add_argument(
        "--seed",
        type=int,
        default=SEED
    )

    # Resume / transfer learning
    ap.add_argument(
        "--resume",
        default=None,
        help="Path to last.pt to resume an interrupted training run"
    )

    ap.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Freeze first N layers"
    )

    ap.add_argument(
        "--cache",
        action="store_true",
        help="Cache images in RAM"
    )

    ap.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch instead of using pretrained weights"
    )

    # -----------------------------------------------------------------------
    # DATA AUGMENTATION
    # -----------------------------------------------------------------------

    ap.add_argument(
        "--hsv-h",
        type=float,
        default=0.015,
        help="Hue augmentation"
    )

    ap.add_argument(
        "--hsv-s",
        type=float,
        default=0.50,
        help="Saturation augmentation"
    )

    ap.add_argument(
        "--hsv-v",
        type=float,
        default=0.40,
        help="Brightness/value augmentation"
    )

    ap.add_argument(
        "--degrees",
        type=float,
        default=5.0,
        help="Small rotation in degrees"
    )

    ap.add_argument(
        "--translate",
        type=float,
        default=0.10,
        help="Image translation fraction"
    )

    ap.add_argument(
        "--scale",
        type=float,
        default=0.50,
        help="Image scaling factor"
    )

    ap.add_argument(
        "--shear",
        type=float,
        default=2.0,
        help="Small shear transformation"
    )

    ap.add_argument(
        "--perspective",
        type=float,
        default=0.0005,
        help="Small perspective transformation"
    )

    ap.add_argument(
        "--fliplr",
        type=float,
        default=0.50,
        help="Horizontal flip probability"
    )

    ap.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="Mosaic augmentation probability"
    )

    ap.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Disable Mosaic for the final N epochs"
    )

    return ap.parse_args()


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def preflight(data_path: Path) -> dict:
    """
    Check that the dataset exists, contains classes, and has
    annotated training and validation images.
    """

    if not data_path.exists():
        sys.exit(
            "ERROR: {} not found. Run prepare_dataset.py first."
            .format(data_path)
        )

    cfg = yaml.safe_load(
        data_path.read_text(encoding="utf-8")
    )

    names = cfg.get("names") or {}

    if isinstance(names, list):
        names = {
            i: n
            for i, n in enumerate(names)
        }

    if not names:
        sys.exit(
            "ERROR: data.yaml declares no classes."
        )

    base = (
        data_path.parent /
        cfg.get("path", ".")
    ).resolve()

    summary = {}

    for split in ("train", "val"):

        rel = cfg.get(split)

        if not rel:
            sys.exit(
                "ERROR: data.yaml has no '{}' split."
                .format(split)
            )

        img_dir = (
            base / rel
        ).resolve()

        lbl_dir = Path(
            str(img_dir).replace(
                "images",
                "labels",
                1
            )
        )

        if not img_dir.is_dir():
            sys.exit(
                "ERROR: {} does not exist. "
                "Run prepare_dataset.py first."
                .format(img_dir)
            )

        n_img = sum(
            1
            for p in img_dir.iterdir()
            if p.suffix.lower()
            in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            }
        )

        n_box = 0

        if lbl_dir.is_dir():

            for lp in lbl_dir.glob("*.txt"):

                n_box += sum(
                    1
                    for ln in lp.read_text(
                        errors="ignore"
                    ).splitlines()
                    if ln.strip()
                )

        summary[split] = {
            "images": n_img,
            "boxes": n_box
        }

        if n_img == 0:
            sys.exit(
                "ERROR: no images in {}"
                .format(img_dir)
            )

        if n_box == 0:

            sys.exit(
                "ERROR: {} contains {:,} images "
                "but ZERO bounding boxes.\n"
                "       The dataset is not annotated. "
                "Training would produce a model\n"
                "       that detects nothing. "
                "Annotate first - see README.md."
                .format(split, n_img)
            )

    return {
        "classes": names,
        "splits": summary
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main() -> int:

    import torch
    from ultralytics import YOLO

    args = build_args()

    # Resolve dataset path
    data_path = (
        ROOT / args.data
        if not Path(args.data).is_absolute()
        else Path(args.data)
    )

    # Validate dataset before training
    info = preflight(data_path)

    # Select device automatically
    device = (
        args.device
        or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    # Windows + Ultralytics multiprocessing can sometimes hang.
    workers = (
        args.workers
        if args.workers is not None
        else (
            0
            if platform.system() == "Windows"
            else 8
        )
    )

    # -----------------------------------------------------------------------
    # Print configuration
    # -----------------------------------------------------------------------

    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)

    print(
        "  model        : {}"
        .format(args.model)
    )

    print(
        "  data         : {}"
        .format(data_path)
    )

    print(
        "  classes      : {} -> {}"
        .format(
            len(info["classes"]),
            list(info["classes"].values())
        )
    )

    print(
        "  train        : {:,} images / {:,} boxes"
        .format(
            info["splits"]["train"]["images"],
            info["splits"]["train"]["boxes"]
        )
    )

    print(
        "  val          : {:,} images / {:,} boxes"
        .format(
            info["splits"]["val"]["images"],
            info["splits"]["val"]["boxes"]
        )
    )

    print(
        "  imgsz        : {}"
        .format(args.imgsz)
    )

    print(
        "  epochs       : {}   patience: {}"
        .format(
            args.epochs,
            args.patience
        )
    )

    print(
        "  batch        : {}   workers: {}"
        .format(
            args.batch,
            workers
        )
    )

    print(
        "  optimizer    : {}   lr0: {}  lrf: {}"
        .format(
            args.optimizer,
            args.lr0,
            args.lrf
        )
    )

    print(
        "  device       : {}"
        .format(device)
    )

    if device == "cpu":

        print(
            "               NOTE: CPU training is "
            "20-50x slower."
        )

        print(
            "               For a first sanity check "
            "use --epochs 5 --batch 4 --imgsz 416."
        )

    else:

        print(
            "               GPU: {}"
            .format(
                torch.cuda.get_device_name(0)
            )
        )

    print(
        "  seed         : {}"
        .format(args.seed)
    )

    # -----------------------------------------------------------------------
    # Print augmentation configuration
    # -----------------------------------------------------------------------

    print("-" * 70)
    print("DATA AUGMENTATION")
    print("-" * 70)

    print(
        "  hsv_h        : {}"
        .format(args.hsv_h)
    )

    print(
        "  hsv_s        : {}"
        .format(args.hsv_s)
    )

    print(
        "  hsv_v        : {}"
        .format(args.hsv_v)
    )

    print(
        "  degrees      : {}"
        .format(args.degrees)
    )

    print(
        "  translate    : {}"
        .format(args.translate)
    )

    print(
        "  scale        : {}"
        .format(args.scale)
    )

    print(
        "  shear        : {}"
        .format(args.shear)
    )

    print(
        "  perspective  : {}"
        .format(args.perspective)
    )

    print(
        "  fliplr       : {}"
        .format(args.fliplr)
    )

    print(
        "  mosaic       : {}"
        .format(args.mosaic)
    )

    print(
        "  close_mosaic : {} epochs"
        .format(args.close_mosaic)
    )

    print("=" * 70)

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------

    if args.resume:

        print(
            "\nResuming training from:"
        )

        print(
            "  {}"
            .format(args.resume)
        )

        model = YOLO(args.resume)

    else:

        print(
            "\nLoading model:"
        )

        print(
            "  {}"
            .format(args.model)
        )

        model = YOLO(args.model)

    # -----------------------------------------------------------------------
    # Resolve dataset path for Ultralytics
    # -----------------------------------------------------------------------

    resolved_data = resolve_for_ultralytics(
        data_path
    )

    # -----------------------------------------------------------------------
    # Training configuration
    # -----------------------------------------------------------------------

    train_kwargs = dict(

        data=str(resolved_data),

        epochs=args.epochs,

        imgsz=args.imgsz,

        batch=args.batch,

        patience=args.patience,

        device=device,

        workers=workers,

        # Use an absolute output directory.
        project=str(
            (
                ROOT / args.project
            ).resolve()
            if not Path(
                args.project
            ).is_absolute()
            else args.project
        ),

        name=args.name,

        seed=args.seed,

        pretrained=not args.no_pretrained,

        optimizer=args.optimizer,

        lr0=args.lr0,

        lrf=args.lrf,

        cache=args.cache,

        # Save evaluation plots including
        # confusion matrix and PR curves.
        plots=True,

        save=True,

        save_period=-1,

        exist_ok=True,

        resume=bool(args.resume),

        # -------------------------------------------------------------------
        # DATA AUGMENTATION
        # -------------------------------------------------------------------

        hsv_h=args.hsv_h,

        hsv_s=args.hsv_s,

        hsv_v=args.hsv_v,

        degrees=args.degrees,

        translate=args.translate,

        scale=args.scale,

        shear=args.shear,

        perspective=args.perspective,

        fliplr=args.fliplr,

        mosaic=args.mosaic,

        close_mosaic=args.close_mosaic,

        # Validation enabled
        val=True,
    )

    # Optional layer freezing
    if args.freeze is not None:

        train_kwargs["freeze"] = args.freeze

    # -----------------------------------------------------------------------
    # Start training
    # -----------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    results = model.train(
        **train_kwargs
    )

    # -----------------------------------------------------------------------
    # Save training information
    # -----------------------------------------------------------------------

    save_dir = Path(
        getattr(
            results,
            "save_dir",
            Path(args.project) / args.name
        )
    )

    record = {

        "timestamp":
            datetime.now().isoformat(
                timespec="seconds"
            ),

        "hyperparameters": {
            k: (
                str(v)
                if isinstance(v, Path)
                else v
            )
            for k, v in train_kwargs.items()
        },

        "dataset":
            info["splits"],

        "classes": {
            str(k): v
            for k, v in info["classes"].items()
        },

        "environment": {

            "python":
                sys.version.split()[0],

            "torch":
                torch.__version__,

            "cuda_available":
                torch.cuda.is_available(),

            "gpu":
                (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),

            "platform":
                platform.platform(),
        },
    }

    (
        save_dir /
        "hyperparameters.json"
    ).write_text(
        json.dumps(
            record,
            indent=2
        ),
        encoding="utf-8"
    )

    # -----------------------------------------------------------------------
    # Final output
    # -----------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print(
        "  best weights : {}"
        .format(
            save_dir /
            "weights" /
            "best.pt"
        )
    )

    print(
        "  last weights : {}"
        .format(
            save_dir /
            "weights" /
            "last.pt"
        )
    )

    print(
        "  plots/metrics: {}"
        .format(save_dir)
    )

    print(
        "  hyperparams  : {}"
        .format(
            save_dir /
            "hyperparameters.json"
        )
    )

    print(
        "\nNext: python scripts/evaluate.py "
        "--weights {}"
        .format(
            save_dir /
            "weights" /
            "best.pt"
        )
    )

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())