#!/usr/bin/env python3
"""
media.py
========
Run the SAME detection + navigation chain over an uploaded image or video,
instead of the live camera.

Why this exists
---------------
A live demo depends on having something worth pointing a camera at. For a
thesis defence you also need a repeatable run: the same file, the same boxes,
the same spoken instruction, every time. This module gives that without
duplicating any logic - it reuses the pipeline's own detector, spatial
analyser and navigation engine, so what you see here is what the live system
would have said about those frames.

    upload  ->  decode  ->  detector  ->  spatial  ->  navigation  ->  result
                                                          |
                                                          +-> spoken text

Images produce one annotated JPEG plus one instruction.
Videos are SAMPLED (default ~2 frames/second, capped) rather than processed
frame-by-frame: a 60-second clip at 30 fps is 1,800 frames, and a blind user
does not need 1,800 instructions - they need the sequence of distinct events.
Consecutive duplicate instructions are collapsed for the same reason the live
system has a cooldown.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

LOG = logging.getLogger("media")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}

# A video is sampled, not exhaustively decoded. These bound the work so a long
# upload cannot hang the server.
SAMPLE_FPS = 2.0        # frames analysed per second of video
MAX_FRAMES = 90         # hard ceiling regardless of length


def kind_of(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return "unsupported"


class MediaAnalyzer:
    """
    Borrows the live pipeline's components so uploaded media is analysed by
    exactly the same model, thresholds and navigation rules as the camera.
    """

    def __init__(self, pipeline, output_dir: Path):
        self.pipeline = pipeline
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def _annotate(self, frame, detections):
        edges = self.pipeline.spatial.zone_boundaries_px(frame.shape[1])
        return self.pipeline.detector.draw_detections(frame, detections, zone_edges=edges)

    def _describe(self, detections, language: str) -> dict:
        """Detections -> the instruction the navigation engine would give."""
        from voice.message_generator import generate_multi_message, clear_path_message

        nav = self.pipeline.navigator
        relevant = nav._relevant(detections)
        if not relevant:
            return {"text": clear_path_message(language), "items": [], "urgent": False}

        ranked = sorted(relevant, key=nav.priority_score, reverse=True)
        n = int(self.pipeline.settings.get("max_spoken_objects"))
        items = [{
            "class_name": d["class_name"],
            "position": d.get("position", "CENTER"),
            "proximity": d.get("proximity", "MEDIUM"),
            "confidence": d.get("confidence", 0.0),
            "urgent": nav._is_urgent(d),
        } for d in ranked[:n]]

        return {
            "text": generate_multi_message(items, language=language,
                                           bn_overrides=nav.bn_overrides),
            "items": items,
            "urgent": bool(items and items[0]["urgent"]),
        }

    def _detect(self, frame):
        h, w = frame.shape[:2]
        dets = self.pipeline.detector.detect(frame)
        for d in dets:
            self.pipeline.spatial.annotate(d, w, h)
        return dets

    # ------------------------------------------------------------------ #
    def analyze_frame(self, frame, language: str, *, live: bool = False) -> dict:
        """Detect on one already-decoded frame and return JSON only.

        This is the path used by a phone's own camera: the browser grabs a
        frame with getUserMedia, posts it here, and gets back boxes plus an
        instruction. It runs the SAME detector, spatial analyser and
        navigation engine as the live camera, so a phone and the desktop
        cannot disagree about what is in front of them.

        Unlike analyze_image it writes nothing to disk. At 5-10 frames per
        second an annotated JPEG per call would be hundreds of files a
        minute; the browser draws the boxes itself from `bbox_norm`.

        Boxes are returned NORMALISED (0-1). The phone decides how the video
        is letterboxed on screen, so pixel coordinates computed here would be
        wrong the moment the display size differs from the capture size.
        """
        import time as _time

        h, w = frame.shape[:2]
        t0 = _time.perf_counter()
        detections = self._detect(frame)
        elapsed = (_time.perf_counter() - t0) * 1000

        # A one-shot analysis just describes the scene. A live stream must also
        # decide whether the scene is NEWS: generate_guidance applies the same
        # priority, event detection and cooldown the desktop camera uses, so a
        # phone does not repeat "chair ahead" ten times a second.
        if live:
            decision = self.pipeline.navigator.generate_guidance(
                detections, language=language)
            guidance = {
                "text": decision.get("text", ""),
                "items": decision.get("items", []),
                "urgent": bool(decision.get("urgent")),
            }
            should_speak = bool(decision.get("speak"))
            reason = decision.get("reason", "")
        else:
            guidance = self._describe(detections, language)
            should_speak = bool(guidance["text"])
            reason = "one_shot"

        return {
            "ok": True,
            "speak": should_speak,
            "reason": reason,
            "width": w,
            "height": h,
            "inference_ms": round(elapsed, 1),
            "object_count": len(detections),
            "detections": [
                {
                    "class_name": d["class_name"],
                    "confidence": d["confidence"],
                    "position": d.get("position"),
                    "proximity": d.get("proximity"),
                    "source": d.get("source", "thesis"),
                    "bbox_norm": d["bbox_norm"],
                }
                for d in detections
            ],
            "guidance": guidance["text"],
            "urgent": guidance["urgent"],
            "items": guidance["items"],
        }

    # ------------------------------------------------------------------ #
    def analyze_image(self, path: Path, language: str) -> dict:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(
                "Could not decode '{}'. The file may be corrupt or not a real "
                "image.".format(path.name))

        t0 = time.perf_counter()
        detections = self._detect(frame)
        elapsed = (time.perf_counter() - t0) * 1000

        guidance = self._describe(detections, language)
        out_name = "{}.jpg".format(uuid.uuid4().hex[:12])
        cv2.imwrite(str(self.output_dir / out_name),
                    self._annotate(frame, detections),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        return {
            "kind": "image",
            "source_name": path.name,
            "annotated_url": "/static/uploads/" + out_name,
            "resolution": "{}x{}".format(frame.shape[1], frame.shape[0]),
            "inference_ms": round(elapsed, 1),
            "detections": detections,
            "object_count": len(detections),
            "guidance": guidance["text"],
            "urgent": guidance["urgent"],
            "timeline": [],
        }

    # ------------------------------------------------------------------ #
    def analyze_video(self, path: Path, language: str) -> dict:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(
                "Could not open '{}'. The codec may be unsupported.".format(path.name))

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not (1.0 <= src_fps <= 240.0):
            src_fps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        step = max(1, int(round(src_fps / SAMPLE_FPS)))
        planned = (total // step) if total else MAX_FRAMES
        if planned > MAX_FRAMES and total:
            step = max(1, total // MAX_FRAMES)

        timeline: list[dict] = []
        first_annotated = None
        analysed = 0
        idx = 0
        counts: dict[str, int] = {}
        conf_sum = 0.0
        conf_n = 0
        t0 = time.perf_counter()

        while analysed < MAX_FRAMES:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % step:
                idx += 1
                continue
            idx += 1

            detections = self._detect(frame)
            guidance = self._describe(detections, language)
            analysed += 1

            for d in detections:
                counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
                conf_sum += float(d.get("confidence", 0.0))
                conf_n += 1

            if first_annotated is None:
                first_annotated = self._annotate(frame, detections)

            t_sec = round(idx / src_fps, 2)
            # Collapse repeats: the same sentence held for 20 frames is one
            # event, not twenty. Same reasoning as the live cooldown.
            if not timeline or timeline[-1]["text"] != guidance["text"]:
                timeline.append({
                    "t": t_sec,
                    "text": guidance["text"],
                    "urgent": guidance["urgent"],
                    "objects": [{"class_name": d["class_name"],
                                 "position": d.get("position"),
                                 "proximity": d.get("proximity"),
                                 "confidence": d.get("confidence")}
                                for d in detections[:4]],
                })
        cap.release()

        if analysed == 0:
            raise ValueError("No frames could be read from '{}'.".format(path.name))

        out_name = "{}.jpg".format(uuid.uuid4().hex[:12])
        if first_annotated is not None:
            cv2.imwrite(str(self.output_dir / out_name), first_annotated,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        return {
            "kind": "video",
            "source_name": path.name,
            "annotated_url": "/static/uploads/" + out_name,
            "duration_s": round(total / src_fps, 1) if total else None,
            "source_fps": round(src_fps, 1),
            "frames_analysed": analysed,
            "sample_step": step,
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "object_count": sum(counts.values()),
            "class_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "avg_confidence": round(conf_sum / conf_n, 4) if conf_n else None,
            "detections": [],
            "guidance": timeline[-1]["text"] if timeline else "",
            "urgent": bool(timeline and timeline[-1]["urgent"]),
            "timeline": timeline,
        }

    # ------------------------------------------------------------------ #
    def analyze(self, path: Path, language: str) -> dict:
        k = kind_of(path.name)
        if k == "image":
            return self.analyze_image(path, language)
        if k == "video":
            return self.analyze_video(path, language)
        raise ValueError(
            "Unsupported file type '{}'. Upload an image ({}) or a video ({})."
            .format(Path(path.name).suffix or "?",
                    ", ".join(sorted(IMAGE_EXTS)), ", ".join(sorted(VIDEO_EXTS))))
