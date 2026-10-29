#!/usr/bin/env python3
"""
detection_core.py
=================
The single shared detection wrapper used by inference.py and (later) by the
navigation module. Everything that touches the YOLO model goes through here so
the navigation/TTS stage never has to know about Ultralytics internals.

Design goal (thesis section 18-19): one function turns a frame into a plain list
of dictionaries. Those dictionaries are the ONLY contract the navigation layer
depends on.

    detections = [
        {
            "class_name": "chair", "class_id": 2, "confidence": 0.91,
            "x_center": 320.0, "y_center": 280.0, "width": 120.0, "height": 200.0,
            "x1": 260.0, "y1": 180.0, "x2": 380.0, "y2": 380.0,
            "area_ratio": 0.078,          # box area / frame area
            "zone": "CENTER",             # LEFT | CENTER | RIGHT
            "proximity": "NEAR",          # NEAR | MEDIUM | FAR  (see caveat below)
        }, ...
    ]

PROXIMITY CAVEAT - important for the viva
-----------------------------------------
`proximity` here is a MONOCULAR HEURISTIC based on apparent box size, not a
measured distance. A large box usually means a close object, but a bus 30 m away
can occupy more pixels than a chair 2 m away. It is a placeholder so the
navigation module has a field to consume. Real distance requires one of:
  - a monocular depth model (e.g. MiDaS / Depth-Anything),
  - a stereo camera or a phone's ToF/LiDAR sensor,
  - or camera-height + ground-plane geometry for objects touching the floor.
Do NOT report this heuristic as "distance estimation" in the thesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class DetectorConfig:
    weights: str = "runs/train/exp/weights/best.pt"
    conf: float = 0.25            # confidence threshold
    iou: float = 0.45             # NMS IoU threshold
    imgsz: int = 640
    device: str | None = None     # None -> auto (cuda if available, else cpu)
    max_det: int = 50
    half: bool = False            # fp16; only meaningful on CUDA

    # LEFT/CENTER/RIGHT split points as fractions of frame width.
    # 0.33/0.66 gives three equal columns. Widening CENTER (e.g. 0.40/0.60)
    # makes the system warn about head-on obstacles more eagerly.
    zone_left_edge: float = 0.33
    zone_right_edge: float = 0.66

    # NEAR/MEDIUM/FAR thresholds on box-area / frame-area.
    near_area_ratio: float = 0.15
    medium_area_ratio: float = 0.04

    classes: list[int] | None = None  # restrict to a subset of class ids


def select_device(preferred: str | None = None) -> str:
    """Auto-select CUDA when available; never require a GPU."""
    import torch

    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Spatial reasoning helpers (consumed later by the navigation module)
# --------------------------------------------------------------------------- #
def horizontal_zone(x_center: float, frame_w: int, cfg: DetectorConfig) -> str:
    f = x_center / max(frame_w, 1)
    if f < cfg.zone_left_edge:
        return "LEFT"
    if f > cfg.zone_right_edge:
        return "RIGHT"
    return "CENTER"


def proximity_bucket(area_ratio: float, cfg: DetectorConfig) -> str:
    if area_ratio >= cfg.near_area_ratio:
        return "NEAR"
    if area_ratio >= cfg.medium_area_ratio:
        return "MEDIUM"
    return "FAR"


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
@dataclass
class Detector:
    cfg: DetectorConfig
    model: object = field(init=False, default=None)
    names: dict = field(init=False, default_factory=dict)
    device: str = field(init=False, default="cpu")

    def __post_init__(self) -> None:
        from ultralytics import YOLO

        wpath = Path(self.cfg.weights)
        if not wpath.exists():
            raise FileNotFoundError(
                "Model weights not found: {}\n"
                "Train a model first (python scripts/train.py) or pass "
                "--weights pointing at an existing .pt file.".format(wpath)
            )
        self.device = select_device(self.cfg.device)
        self.model = YOLO(str(wpath))
        self.names = self.model.names
        print("[detector] weights={}  device={}  classes={}".format(
            wpath.name, self.device, len(self.names)))

    # ------------------------------------------------------------------ #
    def predict(self, frame: np.ndarray) -> list[dict]:
        """Run detection on one BGR frame and return the structured contract."""
        h, w = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            imgsz=self.cfg.imgsz,
            device=self.device,
            max_det=self.cfg.max_det,
            classes=self.cfg.classes,
            verbose=False,
            **self._precision_kwargs(),
        )
        return self._to_records(results[0], w, h)

    # ------------------------------------------------------------------ #
    def _precision_kwargs(self) -> dict:
        """fp16 only helps on CUDA. Ultralytics renamed `half` -> `quantize`
        in 8.4.x, so probe the signature instead of guessing."""
        if not (self.cfg.half and self.device.startswith("cuda")):
            return {}
        import inspect
        try:
            params = inspect.signature(self.model.predict).parameters
        except (TypeError, ValueError):
            return {}
        if "quantize" in params:
            return {"quantize": "half"}
        if "half" in params:
            return {"half": True}
        return {}

    # ------------------------------------------------------------------ #
    def _to_records(self, result, frame_w: int, frame_h: int) -> list[dict]:
        out: list[dict] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        frame_area = float(max(frame_w * frame_h, 1))

        for (x1, y1, x2, y2), cf, ci in zip(xyxy, confs, clss):
            bw, bh = float(x2 - x1), float(y2 - y1)
            cx, cy = float(x1) + bw / 2.0, float(y1) + bh / 2.0
            area_ratio = (bw * bh) / frame_area
            out.append({
                "class_name": self.names.get(int(ci), str(int(ci))),
                "class_id": int(ci),
                "confidence": round(float(cf), 4),
                "x_center": round(cx, 1),
                "y_center": round(cy, 1),
                "width": round(bw, 1),
                "height": round(bh, 1),
                "x1": round(float(x1), 1),
                "y1": round(float(y1), 1),
                "x2": round(float(x2), 1),
                "y2": round(float(y2), 1),
                "area_ratio": round(float(area_ratio), 5),
                "zone": horizontal_zone(cx, frame_w, self.cfg),
                "proximity": proximity_bucket(area_ratio, self.cfg),
            })

        # Most navigation-relevant first: closest, then most confident.
        out.sort(key=lambda d: (-d["area_ratio"], -d["confidence"]))
        return out


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
_PALETTE = [
    (56, 168, 255), (99, 214, 116), (255, 176, 59), (233, 92, 122),
    (167, 128, 255), (64, 214, 214), (255, 128, 64), (140, 200, 90),
]


def draw_detections(frame: np.ndarray, detections: list[dict],
                    show_zone: bool = True, cfg: DetectorConfig | None = None) -> np.ndarray:
    """Draw boxes, class names and confidences onto a copy of the frame."""
    import cv2

    img = frame.copy()
    h, w = img.shape[:2]

    if show_zone and cfg is not None:
        for f in (cfg.zone_left_edge, cfg.zone_right_edge):
            x = int(f * w)
            cv2.line(img, (x, 0), (x, h), (70, 70, 70), 1, cv2.LINE_AA)

    for d in detections:
        color = _PALETTE[d["class_id"] % len(_PALETTE)]
        p1 = (int(d["x1"]), int(d["y1"]))
        p2 = (int(d["x2"]), int(d["y2"]))
        cv2.rectangle(img, p1, p2, color, 2, cv2.LINE_AA)

        # Scale text with frame width so labels stay readable at 256px and
        # do not swamp the image; drop the zone tag when there is no room.
        scale = max(0.32, min(0.6, w / 1100.0))
        label = "{} {:.2f}".format(d["class_name"], d["confidence"])
        if show_zone and w >= 480:
            label += " [{}/{}]".format(d["zone"], d["proximity"])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        tx = min(p1[0], max(0, w - tw - 7))
        ty = max(p1[1], th + 6)
        cv2.rectangle(img, (tx, ty - th - 6), (tx + tw + 6, ty), color, -1)
        cv2.putText(img, label, (tx + 3, ty - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
# Navigation hand-off stub (NOT the navigation system - just the seam)
# --------------------------------------------------------------------------- #
def summarise_for_navigation(detections: list[dict], top_k: int = 3) -> list[dict]:
    """
    Reduce a full detection list to the few objects a navigation message would
    mention. The actual sentence generation and TTS belong to a later module;
    this only defines the hand-off shape so that module can be dropped in.
    """
    priority = {"NEAR": 0, "MEDIUM": 1, "FAR": 2}
    ranked = sorted(detections, key=lambda d: (priority[d["proximity"]], -d["confidence"]))
    return [
        {"class_name": d["class_name"], "zone": d["zone"],
         "proximity": d["proximity"], "confidence": d["confidence"]}
        for d in ranked[:top_k]
    ]
