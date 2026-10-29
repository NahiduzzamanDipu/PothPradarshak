#!/usr/bin/env python3
"""
pipeline.py
===========
The worker thread that runs the whole vision chain:

    camera frame -> YOLO -> spatial analysis -> navigation -> voice
                        \\-> annotated JPEG for the MJPEG stream

Runs in ONE background thread. FastAPI request handlers never do vision work;
they only read the latest published state under a lock. That keeps the HTTP
layer responsive no matter how slow inference is, and means several browser
tabs can watch the same stream without multiplying the GPU load.

Frame flow and why frames are dropped
-------------------------------------
The camera thread always holds the newest frame. This loop takes whatever is
current when it is ready for work. If inference takes 80 ms and the camera
delivers at 30 FPS, roughly two frames are skipped each cycle - correct
behaviour for a navigation aid, where a fresh frame matters far more than
processing every frame.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

import cv2

from camera import CameraSource, CameraError
from detector import ObjectDetector
from spatial import SpatialAnalyzer
from navigation import NavigationEngine
from voice import VoiceEngine

LOG = logging.getLogger("pipeline")


class DetectionPipeline:
    """Owns the camera, the model and the worker thread."""

    def __init__(self, settings, model_info, bn_overrides: dict | None = None):
        self.settings = settings
        self.model_info = model_info

        self.detector = ObjectDetector(model_info, settings)
        self.spatial = SpatialAnalyzer(settings)
        self.navigator = NavigationEngine(settings, bn_overrides=bn_overrides)
        self.voice = VoiceEngine(settings)
        self.camera: CameraSource | None = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()

        # ---- published state (read by the API under _lock) ----
        self._jpeg: bytes | None = None
        self._detections: list[dict] = []
        self._guidance: dict = {"text": "", "speak": False, "urgent": False,
                                "reason": "idle", "items": []}
        self._frame_shape = (0, 0)

        self.running = False
        self.last_error: str | None = None
        self.fps = 0.0
        self.inference_ms = 0.0
        self.frames_processed = 0
        self.started_at: float | None = None
        self._fps_hist = deque(maxlen=30)

        self.voice.start()

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Open the camera and start processing. Raises CameraError on failure."""
        with self._lock:
            if self.running:
                LOG.info("Pipeline already running")
                return

            source = self.settings.get("camera_source")
            cam = CameraSource(
                source=source,
                width=self.settings.get("capture_width"),
                height=self.settings.get("capture_height"),
                loop=bool(self.settings.get("video_loop")),
            )
            cam.open()                      # raises CameraError
            self.camera = cam

            self.navigator.reset()
            self._stop.clear()
            self.last_error = None
            self.frames_processed = 0
            self._fps_hist.clear()
            self.started_at = time.time()
            self.running = True

        self._thread = threading.Thread(target=self._loop, name="pipeline", daemon=True)
        self._thread.start()
        LOG.info("Detection started.")

    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        LOG.info("Stopping detection...")
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=3.0)
        self._thread = None

        with self._lock:
            if self.camera is not None:
                self.camera.close()
                self.camera = None
            self.running = False
            self._detections = []
            self._guidance = {"text": "", "speak": False, "urgent": False,
                              "reason": "stopped", "items": []}
        LOG.info("Detection stopped.")

    # ------------------------------------------------------------------ #
    def restart_camera(self) -> None:
        """Reopen with new camera settings, preserving running/stopped state."""
        was = self.running
        self.stop()
        if was:
            self.start()

    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        self.stop()
        self.voice.stop()

    # ------------------------------------------------------------------ #
    def _loop(self) -> None:
        last_seq = -1
        skip_counter = 0
        last_detections: list[dict] = []

        while not self._stop.is_set():
            cam = self.camera
            if cam is None:
                break

            ok, frame, seq = cam.read_new(last_seq)
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            if seq == last_seq:
                # No new frame yet; yield rather than re-processing the same one.
                time.sleep(0.003)
                continue
            last_seq = seq

            t0 = time.perf_counter()

            # ---- optional frame skipping (section 22) ---- #
            skip = int(self.settings.get("frame_skip"))
            detect_on = bool(self.settings.get("detection_enabled"))
            run_inference = detect_on and ((skip == 0) or (skip_counter % (skip + 1) == 0))
            skip_counter += 1

            # Detection switched off: stream the raw camera, drop stale boxes.
            if not detect_on:
                last_detections = []

            h, w = frame.shape[:2]
            self._frame_shape = (w, h)

            if run_inference:
                try:
                    detections = self.detector.detect(frame)
                except Exception as exc:
                    LOG.error("Detection error: %s", exc)
                    self.last_error = str(exc)
                    detections = []

                for d in detections:
                    self.spatial.annotate(d, w, h)
                last_detections = detections
                self.inference_ms = self.detector.last_inference_ms
            else:
                # Reuse the previous boxes so the overlay does not flicker.
                # (Empty when detection is disabled - see above.)
                detections = last_detections

            # ---- navigation + voice ---- #
            if run_inference:
                try:
                    guidance = self.navigator.generate_guidance(detections)
                except Exception as exc:
                    LOG.error("Navigation error: %s", exc)
                    guidance = {"text": "", "speak": False, "urgent": False,
                                "reason": "error:{}".format(exc), "items": []}

                if guidance.get("speak") and guidance.get("text"):
                    # Non-blocking: enqueues and returns immediately.
                    self.voice.speak(guidance["text"],
                                     language=self.settings.get("language"),
                                     urgent=guidance.get("urgent", False))
                    LOG.info("SPEAK [%s] reason=%s", self.settings.get("language"),
                             guidance.get("reason"))
            elif not detect_on:
                # Detection is off: say nothing and claim nothing. Holding the
                # last sentence here would assert an obstacle the system is no
                # longer looking for.
                guidance = {"text": "", "speak": False, "urgent": False,
                            "reason": "detection_disabled", "items": []}
                self.navigator.reset()
            else:
                guidance = self._guidance

            # ---- draw overlay ---- #
            try:
                edges = self.spatial.zone_boundaries_px(w)
                annotated = self.detector.draw_detections(frame, detections, zone_edges=edges)
                banner = ["FPS {:.1f} | {} obj | {} | {:.0f} ms".format(
                    self.fps, len(detections), self.detector.device, self.inference_ms)]
                if not detect_on:
                    banner.append("! DETECTION OFF")
                if self.model_info.is_placeholder:
                    banner.append("! PLACEHOLDER MODEL (COCO) - not the thesis model")
                annotated = self.detector.draw_banner(annotated, banner)

                ok_enc, buf = cv2.imencode(
                    ".jpg", annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.settings.get("jpeg_quality"))])
                jpeg = buf.tobytes() if ok_enc else None
            except Exception as exc:
                LOG.error("Rendering error: %s", exc)
                jpeg = None

            # ---- publish ---- #
            with self._lock:
                if jpeg is not None:
                    self._jpeg = jpeg
                self._detections = detections
                self._guidance = guidance

            self.frames_processed += 1
            dt = time.perf_counter() - t0
            if dt > 0:
                self._fps_hist.append(1.0 / dt)
                self.fps = sum(self._fps_hist) / len(self._fps_hist)

        LOG.debug("Pipeline loop exited")

    # ------------------------------------------------------------------ #
    # Read-only accessors used by the API
    # ------------------------------------------------------------------ #
    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def latest_detections(self) -> list[dict]:
        with self._lock:
            return list(self._detections)

    def latest_guidance(self) -> dict:
        with self._lock:
            return dict(self._guidance)

    # ------------------------------------------------------------------ #
    # Playback control for an uploaded video, forwarded to the capture layer.
    # Kept here so the API talks to the pipeline only, and a missing camera is
    # a normal answer rather than an AttributeError.
    # ------------------------------------------------------------------ #
    def playback_command(self, action: str) -> dict:
        cam = self.camera
        if cam is None:
            return {"ok": False, "error": "No source is open. Press Start first."}
        if not cam.is_file:
            return {"ok": False,
                    "error": "Playback control applies to an uploaded video, "
                             "not to a live camera."}

        if action == "pause":
            cam.pause()
        elif action in ("play", "resume"):
            cam.resume()
        elif action == "restart":
            cam.restart()
            self.navigator.reset()      # a replay is a new scene, not a continuation
        else:
            return {"ok": False, "error": "Unknown action '{}'.".format(action)}
        return {"ok": True, "playback": cam.playback()}

    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        cam_status = self.camera.status() if self.camera else {
            "source": self.settings.get("camera_source"),
            "connected": False, "resolution": None,
            "capture_fps": 0.0, "frames_read": 0, "error": None,
        }
        uptime = round(time.time() - self.started_at, 1) if self.started_at and self.running else 0.0
        return {
            "running": self.running,
            "fps": round(self.fps, 1),
            "inference_ms": round(self.inference_ms, 1),
            "frames_processed": self.frames_processed,
            "uptime_s": uptime,
            "frame_size": "{}x{}".format(*self._frame_shape) if self._frame_shape[0] else None,
            "error": self.last_error,
            "camera": cam_status,
            "model": self.detector.status(),
            "voice": self.voice.status(),
            "navigation": self.navigator.status(),
        }
