#!/usr/bin/env python3
"""
data_utils.py
=============
Shared helper for handing data.yaml to Ultralytics.

WHY THIS EXISTS
---------------
Ultralytics resolves a RELATIVE `path:` in data.yaml against its own global
`datasets_dir` setting (on Windows typically
`C:/Users/<you>/AppData/Roaming/Ultralytics/settings.json`), NOT against the
location of data.yaml itself. So a perfectly sensible

    path: dataset
    train: images/train

silently becomes `<datasets_dir>/dataset/images/train` and training dies with
"images not found". It also mutates behaviour depending on a global file outside
the project, which breaks reproducibility.

This module writes a RESOLVED COPY of data.yaml with an absolute `path:` and
only the keys Ultralytics expects, then hands that copy over. The original
data.yaml stays human-readable, relative and portable - and keeps our extra
documentation keys (`names_bn`, `dataset_provenance`) which are for the thesis
and the navigation module, not for YOLO.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Keys Ultralytics actually reads. Everything else in our data.yaml is
# documentation and must be stripped from the copy handed to it.
_ULTRALYTICS_KEYS = {"path", "train", "val", "test", "names", "nc", "download", "kpt_shape", "flip_idx"}


def load_data_yaml(data_path: str | Path) -> dict:
    """Read data.yaml and normalise `names` to a {int: str} dict."""
    p = Path(data_path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    names = cfg.get("names") or {}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    cfg["names"] = {int(k): str(v) for k, v in names.items()}
    return cfg


def dataset_root(data_path: str | Path, cfg: dict | None = None) -> Path:
    """Absolute dataset root, resolved relative to data.yaml's own folder."""
    p = Path(data_path).resolve()
    cfg = cfg or load_data_yaml(p)
    return (p.parent / cfg.get("path", ".")).resolve()


def labels_dir_for(images_dir: Path) -> Path:
    """YOLO convention: the labels folder mirrors the images folder."""
    s = str(images_dir)
    return Path(s.replace("images", "labels", 1)) if "images" in s else images_dir.parent / "labels"


def resolve_for_ultralytics(data_path: str | Path, out_name: str = ".data.resolved.yaml") -> Path:
    """
    Write a sibling copy of data.yaml with an ABSOLUTE `path:` and only the keys
    Ultralytics understands, and return its path. Safe to call repeatedly.
    """
    p = Path(data_path).resolve()
    cfg = load_data_yaml(p)
    root = dataset_root(p, cfg)

    resolved = {k: v for k, v in cfg.items() if k in _ULTRALYTICS_KEYS}
    resolved["path"] = str(root).replace("\\", "/")

    dest = p.parent / out_name
    header = (
        "# AUTO-GENERATED from {} - do not edit, do not commit.\n"
        "# Ultralytics resolves a relative `path:` against its global datasets_dir,\n"
        "# not against data.yaml, so this copy pins an absolute path.\n"
        "# Regenerated automatically on every train/evaluate run.\n"
    ).format(p.name)
    dest.write_text(header + yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return dest
