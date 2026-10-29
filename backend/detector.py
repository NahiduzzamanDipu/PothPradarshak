#!/usr/bin/env python3
"""
detector.py
===========
The only module that touches Ultralytics. Loads the trained model (or the
flagged placeholder), runs inference, and draws the overlay.

Returns detections in the structure the rest of the app agreed on:

    {
        "class_name": "rickshaw", "class_id": 0, "confidence": 0.91,
        "bbox": [120, 180, 350, 500],          # pixels, x1 y1 x2 y2
        "center": [235, 340],                  # pixels
        "width": 230, "height": 320,
        "bbox_norm": [0.19, 0.33, 0.55, 0.93], # normalised, for the frontend
    }

`position`, `proximity` and `area_ratio` are added afterwards by SpatialAnalyzer
- the detector deliberately knows nothing about navigation.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from config import PLACEHOLDER_MODEL, ModelInfo, resolve_device

LOG = logging.getLogger("detector")

# ---------------------------------------------------------------------------
# Overlay colours (BGR, because OpenCV).
#
# The brief asks for a RED detection style matching the supplied UI design, so
# red is the standard box colour rather than a per-class palette: the class is
# already named in the label, and one colour keeps a busy footpath scene
# readable. Only the genuinely urgent case is styled differently, so that
# "thicker + brighter" always means the same thing to the viewer.
#
# Hex values are taken from the design's own tokens:
#   #ef4444  danger red     -> standard box
#   #dc2626  deeper red     -> NEAR + CENTER (urgent)
#   #f8fafc  near-white     -> label text
# ---------------------------------------------------------------------------
BOX_COLOR = (68, 68, 239)       # #ef4444
DANGER_COLOR = (38, 38, 220)    # #dc2626
LABEL_TEXT = (250, 250, 248)    # #f8fafc
ZONE_LINE = (80, 72, 64)        # faint divider, stays out of the way


# ---------------------------------------------------------------------------
# INDOOR DETECTION - secondary model
# ---------------------------------------------------------------------------
# The thesis model is trained on Dhaka footpaths and knows nothing about
# furniture, so indoors it correctly reports an empty scene. Rather than
# pretend otherwise, a second COCO-pretrained model runs alongside it and
# contributes ONLY the indoor obstacle classes listed here.
#
# This is model fusion, not a replacement: the thesis model remains the
# authority on every outdoor class, and the two never compete for the same
# label because this list deliberately excludes everything the thesis model
# already covers (notably `person`, which both models can see).
#
# Honest limitation for the write-up: these indoor classes are COCO-pretrained,
# NOT trained on thesis data, so they carry no thesis-measured mAP. `door` is
# absent because COCO has no door class - it would need its own annotation set.
#
# Ids are offset by INDOOR_ID_BASE so they can never collide with the thesis
# class ids (0-8) when colouring boxes.
INDOOR_ID_BASE = 100

INDOOR_CLASSES = {
    "chair", "couch", "bed", "dining table", "tv", "toilet",
    "sink", "refrigerator", "oven", "microwave", "potted plant", "bench",
}


class ModelLoadError(RuntimeError):
    pass


class ObjectDetector:
    """
    Wraps a YOLO model.

        det = ObjectDetector(model_info, settings)
        detections = det.detect(frame)
        annotated  = det.draw_detections(frame, detections)
    """

    def __init__(self, model_info: ModelInfo, settings):
        self.info = model_info
        self.settings = settings
        self.model = None
        self.device = "cpu"
        self.class_names: dict = {}
        self.warmup_ms = 0.0
        self.last_inference_ms = 0.0
        # Secondary COCO model for indoor furniture (see INDOOR_CLASSES).
        self.indoor_model = None
        self.indoor_names: dict = {}
        self.indoor_error: str | None = None
        self._load()
        self._load_indoor()

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise ModelLoadError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self.device = resolve_device(self.settings.get("device"))
        LOG.info("Loading YOLO model: %s", self.info.path)
        LOG.info("Model source: %s", self.info.source)

        try:
            self.model = YOLO(self.info.path)
        except Exception as exc:
            raise ModelLoadError(
                "Failed to load model '{}': {}\n"
                "If this is a path to your trained weights, check the file is a "
                "complete .pt produced by Ultralytics.".format(self.info.path, exc)
            ) from exc

        self.class_names = {int(k): str(v) for k, v in self.model.names.items()}
        self.info.class_names = self.class_names
        LOG.info("Model loaded successfully. %d classes, device=%s",
                 len(self.class_names), self.device)

        if self.info.is_placeholder:
            LOG.warning("=" * 70)
            LOG.warning("PLACEHOLDER MODEL IN USE - THIS IS NOT YOUR THESIS MODEL")
            LOG.warning("%s", self.info.source)
            LOG.warning("Detected classes are the 80 generic COCO classes.")
            LOG.warning("Finish Part 1 (annotate -> train) and drop best.pt into")
            LOG.warning("models/ or runs/train/*/weights/ - it is picked up automatically.")
            LOG.warning("=" * 70)

        self._warmup()

    # ------------------------------------------------------------------ #
    def _load_indoor(self) -> None:
        """
        Load the COCO model used for indoor classes. Failure is NOT fatal:
        indoor support is an enhancement, and losing it must never stop the
        thesis model from running.
        """
        if self.info.is_placeholder:
            # Already running COCO as the main model - a second copy would
            # only duplicate every box.
            LOG.info("Indoor model skipped: the main model is already COCO.")
            return
        try:
            from ultralytics import YOLO
            self.indoor_model = YOLO(PLACEHOLDER_MODEL)
            self.indoor_names = {int(k): str(v) for k, v in self.indoor_model.names.items()}
            LOG.info("Indoor model loaded (%s): contributing %d indoor classes",
                     PLACEHOLDER_MODEL, len(INDOOR_CLASSES))
        except Exception as exc:
            self.indoor_error = str(exc)
            self.indoor_model = None
            LOG.warning("Indoor model unavailable (indoor classes disabled): %s", exc)

    # ------------------------------------------------------------------ #
    def _warmup(self) -> None:
        """First inference allocates buffers and is always slow; do it up front
        so the measured FPS is honest from frame one."""
        imgsz = self.settings.get("imgsz")
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        try:
            self.model.predict(dummy, imgsz=imgsz, device=self.device, verbose=False)
            self.warmup_ms = (time.perf_counter() - t0) * 1000
            LOG.info("Warm-up inference: %.0f ms", self.warmup_ms)
        except Exception as exc:
            LOG.warning("Warm-up failed (continuing anyway): %s", exc)

    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run inference on one BGR frame."""
        if self.model is None or frame is None:
            return []
        s = self.settings
        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        try:
            results = self.model.predict(
                source=frame,
                conf=s.get("confidence"),
                iou=s.get("iou"),
                imgsz=s.get("imgsz"),
                device=self.device,
                max_det=s.get("max_det"),
                verbose=False,
            )
        except Exception as exc:
            LOG.error("Inference failed: %s", exc)
            return []
        out = self._decode(results[0], w, h, self.class_names, indoor=False)

        # ---- indoor classes from the secondary COCO model ---- #
        if self.indoor_model is not None and s.get("indoor_enabled"):
            try:
                ir = self.indoor_model.predict(
                    source=frame,
                    conf=s.get("indoor_conf"),
                    iou=s.get("iou"),
                    imgsz=s.get("imgsz"),
                    device=self.device,
                    max_det=s.get("max_det"),
                    verbose=False,
                )
                indoor = self._decode(ir[0], w, h, self.indoor_names, indoor=True)
                out.extend(self._merge(out, indoor))
            except Exception as exc:
                LOG.error("Indoor inference failed: %s", exc)

        self.last_inference_ms = (time.perf_counter() - t0) * 1000
        return out

    # ------------------------------------------------------------------ #
    def _decode(self, result, w: int, h: int, names: dict, indoor: bool) -> list[dict]:
        """Ultralytics result -> this app's detection dicts."""
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        out: list[dict] = []
        for (x1, y1, x2, y2), cf, ci in zip(xyxy, confs, clss):
            name = names.get(int(ci), str(int(ci)))
            # The secondary model contributes ONLY indoor furniture; everything
            # else it sees is already the thesis model's job.
            if indoor and name not in INDOOR_CLASSES:
                continue
            x1, y1 = float(max(0.0, x1)), float(max(0.0, y1))
            x2, y2 = float(min(w, x2)), float(min(h, y2))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
            out.append({
                "class_id": int(ci) + (INDOOR_ID_BASE if indoor else 0),
                "class_name": name,
                "confidence": round(float(cf), 4),
                "source": "indoor" if indoor else "thesis",
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "center": [round(cx), round(cy)],
                "width": round(bw),
                "height": round(bh),
                "bbox_norm": [round(x1 / w, 4), round(y1 / h, 4),
                              round(x2 / w, 4), round(y2 / h, 4)],
            })
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge(primary: list[dict], secondary: list[dict], iou_thr: float = 0.55) -> list[dict]:
        """
        Keep only the secondary boxes that do not already overlap a thesis box.

        Two models looking at one frame will sometimes outline the same object
        (a bench read as street_obstacle, say). Without this the UI lists it
        twice and the voice announces it twice. The thesis model always wins,
        because it is the one trained on this thesis's data.
        """
        def iou(a, b):
            ax1, ay1, ax2, ay2 = a["bbox"]
            bx1, by1, bx2, by2 = b["bbox"]
            ix = max(0, min(ax2, bx2) - max(ax1, bx1))
            iy = max(0, min(ay2, by2) - max(ay1, by1))
            inter = ix * iy
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            return inter / union if union > 0 else 0.0

        return [d for d in secondary
                if not any(iou(d, p) > iou_thr for p in primary)]

    # ------------------------------------------------------------------ #
    def draw_detections(self, frame: np.ndarray, detections: list[dict],
                        zone_edges: tuple[int, int] | None = None,
                        highlight_key: str | None = None) -> np.ndarray:
        """Overlay boxes, labels and the LEFT/CENTER/RIGHT dividers."""
        img = frame.copy()
        h, w = img.shape[:2]
        scale = max(0.4, min(0.62, w / 1100.0))

        if zone_edges:
            for x in zone_edges:
                cv2.line(img, (int(x), 0), (int(x), h), ZONE_LINE, 1, cv2.LINE_AA)

        for d in detections:
            danger = d.get("position") == "CENTER" and d.get("proximity") == "NEAR"
            color = DANGER_COLOR if danger else BOX_COLOR
            x1, y1, x2, y2 = (int(v) for v in d["bbox"])
            thickness = 3 if danger else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

            # Deliberately short: class + confidence only. Position and
            # proximity are already shown per object in the side panel, and
            # repeating them here produced labels wide enough to overlap each
            # other on a busy street scene. The one exception is the urgent
            # case, which earns a marker because it is what the voice is
            # warning about.
            label = "{} {:.0%}".format(d["class_name"], d["confidence"])
            if danger:
                label = "! " + label

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            tx = min(x1, max(0, w - tw - 8))
            ty = max(y1, th + 8)
            cv2.rectangle(img, (tx, ty - th - 8), (tx + tw + 8, ty), color, -1)
            cv2.putText(img, label, (tx + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, LABEL_TEXT, 1, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------------ #
    def draw_banner(self, frame: np.ndarray, lines: list[str]) -> np.ndarray:
        """Small translucent status strip at the top-left (FPS, warnings)."""
        if not lines:
            return frame
        img = frame
        pad, lh = 8, 20
        box_h = pad * 2 + lh * len(lines)
        box_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                    for t in lines) + pad * 2

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        for i, text in enumerate(lines):
            color = (80, 200, 255) if text.startswith("!") else (140, 255, 170)
            cv2.putText(img, text, (pad, pad + lh * (i + 1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return img

    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        return {
            "path": self.info.path,
            "is_placeholder": self.info.is_placeholder,
            "source": self.info.source,
            "device": self.device,
            "num_classes": len(self.class_names),
            "class_names": list(self.class_names.values()),
            "last_inference_ms": round(self.last_inference_ms, 1),
        }
