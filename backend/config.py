#!/usr/bin/env python3
"""
config.py
=========
Single source of truth for every tunable in the application.

Two layers:
  * Settings   - a thread-safe, runtime-mutable dataclass. The API mutates it
                 while the pipeline thread reads it, so every access goes
                 through a lock.
  * discovery  - locating the trained model and the class metadata WITHOUT
                 hardcoding a path, because Part 1's output location depends on
                 how training was run.

Nothing here imports torch or ultralytics, so it stays cheap to import.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

LOG = logging.getLogger("config")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# --------------------------------------------------------------------------- #
# Model discovery
# --------------------------------------------------------------------------- #
# Searched IN ORDER. The first existing file wins. Covers the standard
# Ultralytics layouts plus the layout used by this project's scripts/train.py.
MODEL_SEARCH_GLOBS = [
    "models/best.pt",
    "runs/train/*/weights/best.pt",
    "runs/detect/*/weights/best.pt",
    "runs/*/weights/best.pt",
    "runs/**/weights/best.pt",
    "**/weights/best.pt",
    "best.pt",
]

# Used only when nothing above is found. This is the stock COCO model, NOT a
# thesis model - the app flags it loudly everywhere.
PLACEHOLDER_MODEL = "yolo11n.pt"


@dataclass
class ModelInfo:
    """Where the weights came from, and whether they are the real thing."""
    path: str
    is_placeholder: bool
    source: str            # human-readable explanation
    class_names: dict = field(default_factory=dict)
    class_names_bn: dict = field(default_factory=dict)


def discover_model(explicit: str | None = None) -> ModelInfo:
    """
    Find the trained model. Never raises - falls back to the COCO placeholder so
    the application is always demonstrable, but marks it as a placeholder.
    """
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            return ModelInfo(str(p), False, "explicitly set via --model / MODEL_PATH")
        LOG.warning("Requested model not found: %s - falling back to discovery", p)

    for pattern in MODEL_SEARCH_GLOBS:
        for hit in sorted(PROJECT_ROOT.glob(pattern)):
            if hit.is_file():
                return ModelInfo(str(hit), False,
                                 "auto-discovered at {}".format(hit.relative_to(PROJECT_ROOT)))

    return ModelInfo(
        PLACEHOLDER_MODEL,
        True,
        "NO TRAINED MODEL FOUND. Using the stock COCO-pretrained {} as a "
        "placeholder so the application can run. This is NOT the thesis model."
        .format(PLACEHOLDER_MODEL),
    )


def load_class_metadata(data_yaml: Path | None = None) -> tuple[dict, dict]:
    """
    Read English + Bengali class names from data.yaml if present.
    Returns ({id: en}, {id: bn}); either may be empty.
    """
    path = data_yaml or (PROJECT_ROOT / "data.yaml")
    if not path.exists():
        return {}, {}
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        LOG.warning("Could not parse %s: %s", path, exc)
        return {}, {}

    def norm(v):
        if isinstance(v, list):
            return {i: str(n) for i, n in enumerate(v)}
        if isinstance(v, dict):
            return {int(k): str(n) for k, n in v.items()}
        return {}

    return norm(cfg.get("names")), norm(cfg.get("names_bn"))


# --------------------------------------------------------------------------- #
# Runtime settings
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    # ---- camera ----
    camera_source: str = "0"          # "0"/"1"/"2" or an http(s)/rtsp URL
    capture_width: int = 960          # request from the driver; may be ignored
    capture_height: int = 540
    # ---- inference ----
    confidence: float = 0.50
    iou: float = 0.45
    imgsz: int = 640
    max_det: int = 30
    detection_enabled: bool = True   # UI "Detection ON/OFF"; camera keeps streaming
    # ---- indoor detection (secondary model) ----
    indoor_enabled: bool = True      # run the COCO model alongside the thesis model
    indoor_conf: float = 0.40        # its own threshold; COCO is confident indoors
    # ---- input mode ----
    input_mode: str = "live"         # live | image | video  (UI selector)
    video_loop: bool = False         # uploaded clips stop at the end by default
    frame_skip: int = 0               # run YOLO on 1 of every (frame_skip+1) frames
    device: str = "auto"              # auto | cuda | cpu
    # ---- spatial (section 9 / 10) ----
    zone_left_edge: float = 0.33      # x fraction: below this -> LEFT
    zone_right_edge: float = 0.66     # above this -> RIGHT
    near_area_ratio: float = 0.15     # box area / frame area
    medium_area_ratio: float = 0.04
    # ---- navigation ----
    voice_enabled: bool = True
    language: str = "bn"              # en | bn - Bengali is the thesis default
    announce_cooldown: float = 3.0    # seconds, per object+position key
    global_min_gap: float = 1.2       # seconds between ANY two announcements
    max_spoken_objects: int = 3       # objects per utterance; they are COMBINED
                                      # into one sentence, not read separately
    announce_far: bool = False        # stay quiet about distant clutter
    # ---- tts ----
    tts_backend: str = "auto"         # auto | server | browser | pyttsx3 | off
    tts_voice_bn: str = "bn-BD-NabanitaNeural"
    tts_rate: int = 180               # pyttsx3 words-per-minute
    # ---- stream ----
    jpeg_quality: int = 70
    stream_fps_cap: int = 20

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    # ------------------------------------------------------------------ #
    _MUTABLE = {
        "camera_source", "capture_width", "capture_height",
        "confidence", "iou", "imgsz", "max_det", "detection_enabled",
        "indoor_enabled", "indoor_conf", "input_mode", "video_loop",
        "frame_skip", "device",
        "zone_left_edge", "zone_right_edge", "near_area_ratio", "medium_area_ratio",
        "voice_enabled", "language", "announce_cooldown", "global_min_gap",
        "max_spoken_objects", "announce_far",
        "tts_backend", "tts_voice_bn", "tts_rate",
        "jpeg_quality", "stream_fps_cap",
    }

    # Fields that require the camera to be reopened when changed.
    CAMERA_FIELDS = {"camera_source", "capture_width", "capture_height"}

    _VALIDATORS = {
        "confidence": lambda v: min(max(float(v), 0.01), 0.99),
        "iou": lambda v: min(max(float(v), 0.05), 0.95),
        "imgsz": lambda v: max(160, min(1280, int(v) // 32 * 32)),
        "max_det": lambda v: max(1, min(300, int(v))),
        "frame_skip": lambda v: max(0, min(10, int(v))),
        "zone_left_edge": lambda v: min(max(float(v), 0.05), 0.49),
        "zone_right_edge": lambda v: min(max(float(v), 0.51), 0.95),
        "near_area_ratio": lambda v: min(max(float(v), 0.01), 0.95),
        "medium_area_ratio": lambda v: min(max(float(v), 0.001), 0.9),
        "announce_cooldown": lambda v: min(max(float(v), 0.0), 30.0),
        "global_min_gap": lambda v: min(max(float(v), 0.0), 15.0),
        "max_spoken_objects": lambda v: max(1, min(3, int(v))),
        "tts_rate": lambda v: max(80, min(320, int(v))),
        "jpeg_quality": lambda v: max(30, min(95, int(v))),
        "stream_fps_cap": lambda v: max(1, min(60, int(v))),
        "capture_width": lambda v: max(160, min(1920, int(v))),
        "capture_height": lambda v: max(120, min(1080, int(v))),
        "language": lambda v: v if v in ("en", "bn") else "en",
        "device": lambda v: v if v in ("auto", "cuda", "cpu") else "auto",
        "tts_backend": lambda v: v if v in ("auto", "server", "browser", "pyttsx3", "off") else "auto",
        "voice_enabled": lambda v: bool(v),
        "detection_enabled": lambda v: bool(v),
        "indoor_enabled": lambda v: bool(v),
        "input_mode": lambda v: v if v in ("live", "image", "video") else "live",
        "video_loop": lambda v: bool(v),
        "indoor_conf": lambda v: min(max(float(v), 0.05), 0.95),
        "announce_far": lambda v: bool(v),
        "camera_source": lambda v: str(v).strip(),
    }

    def snapshot(self) -> dict:
        """
        A plain dict copy, safe to serialise or read without holding the lock.

        Built field-by-field rather than with dataclasses.asdict(): asdict()
        deep-copies every field, and deep-copying the RLock raises
        "TypeError: cannot pickle '_thread.RLock' object".
        """
        with self._lock:
            return {f.name: getattr(self, f.name)
                    for f in fields(self) if not f.name.startswith("_")}

    def get(self, name: str):
        with self._lock:
            return getattr(self, name)

    def update(self, changes: dict) -> tuple[dict, bool]:
        """
        Apply validated changes. Returns (applied, camera_needs_restart).
        Unknown or immutable keys are ignored rather than raising, so a stale
        frontend cannot break the backend.
        """
        applied, restart = {}, False
        with self._lock:
            for key, raw in (changes or {}).items():
                if key not in self._MUTABLE:
                    continue
                try:
                    value = self._VALIDATORS.get(key, lambda v: v)(raw)
                except (TypeError, ValueError):
                    LOG.warning("Rejected bad value for %s: %r", key, raw)
                    continue
                if getattr(self, key) != value:
                    setattr(self, key, value)
                    applied[key] = value
                    if key in self.CAMERA_FIELDS:
                        restart = True

            # keep the zone edges from crossing over
            if self.zone_left_edge >= self.zone_right_edge:
                self.zone_right_edge = min(0.95, self.zone_left_edge + 0.02)
                applied["zone_right_edge"] = self.zone_right_edge
            # NEAR must be a bigger box than MEDIUM
            if self.medium_area_ratio >= self.near_area_ratio:
                self.medium_area_ratio = self.near_area_ratio / 2.0
                applied["medium_area_ratio"] = self.medium_area_ratio
        return applied, restart


def resolve_device(preference: str = "auto") -> str:
    """'auto' -> cuda when genuinely available, else cpu. Never requires a GPU."""
    if preference == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if preference == "cuda":
            LOG.warning("device=cuda requested but CUDA is unavailable - using CPU")
    except Exception as exc:
        LOG.warning("Could not query torch for CUDA: %s", exc)
    return "cpu"
