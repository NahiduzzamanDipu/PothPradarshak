#!/usr/bin/env python3
"""
main.py
=======
FastAPI application: REST API + MJPEG video stream + static frontend.

Run:
    python backend/main.py
    python backend/main.py --camera 1
    python backend/main.py --camera "http://192.168.0.101:8080/video"
    python backend/main.py --model runs/train/exp/weights/best.pt --port 8080
    python backend/main.py --host 0.0.0.0        # reachable from your phone

Then open  http://127.0.0.1:8000

Why MJPEG rather than WebSocket video (section 18):
  MJPEG over HTTP is a multipart stream that <img src="/video_feed"> renders
  natively - no client-side decoding, no reconnection logic, ~30 lines of
  server code. A WebSocket would only pay off if the browser needed to send
  frames UP to the server, or if we wanted binary framing for several streams.
  Neither applies. Detection DATA does go over a separate JSON poll, so the
  numbers stay structured and the video stays dumb.
"""

from __future__ import annotations

import argparse
import uuid
import logging
import sys
import time
from pathlib import Path

# Make sibling modules importable whether run as `python backend/main.py`
# or as `uvicorn backend.main:app`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi import File, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from camera import CameraError, probe_indices
from media import MediaAnalyzer, kind_of
from config import (FRONTEND_DIR, PROJECT_ROOT, Settings, discover_model,
                    load_class_metadata)
from detector import ModelLoadError
from pipeline import DetectionPipeline


# --------------------------------------------------------------------------- #
# Logging - force UTF-8 or Bengali guidance text crashes the Windows console
# (cp1252 cannot encode Bengali; this is a real crash, not a cosmetic issue).
# --------------------------------------------------------------------------- #
def setup_logging(level: str = "INFO") -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


LOG = logging.getLogger("main")

# --------------------------------------------------------------------------- #
# Application state
# --------------------------------------------------------------------------- #
settings = Settings()
STATE: dict = {"pipeline": None, "model_info": None, "startup_error": None}

DISCLAIMER = (
    "Research Prototype: This system provides experimental computer-vision "
    "assistance and should not be relied upon as the sole navigation system "
    "for real-world mobility."
)

app = FastAPI(
    title="Visual Assistance System",
    description="Object detection and navigation assistance for visually impaired individuals "
                "(university thesis prototype). " + DISCLAIMER,
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # prototype on a LAN; tighten for any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = PROJECT_ROOT / "static"
(STATIC_DIR / "audio").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "uploads").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class SettingsPatch(BaseModel):
    camera_source: str | None = None
    confidence: float | None = None
    iou: float | None = None
    imgsz: int | None = None
    frame_skip: int | None = None
    detection_enabled: bool | None = None
    indoor_enabled: bool | None = None
    indoor_conf: float | None = None
    input_mode: str | None = None
    video_loop: bool | None = None
    device: str | None = None
    zone_left_edge: float | None = None
    zone_right_edge: float | None = None
    near_area_ratio: float | None = None
    medium_area_ratio: float | None = None
    voice_enabled: bool | None = None
    language: str | None = None
    announce_cooldown: float | None = None
    global_min_gap: float | None = None
    max_spoken_objects: int | None = None
    announce_far: bool | None = None
    tts_backend: str | None = None
    tts_voice_bn: str | None = None
    tts_rate: int | None = None
    jpeg_quality: int | None = None
    capture_width: int | None = None
    capture_height: int | None = None


class LanguagePatch(BaseModel):
    language: str


class VoicePatch(BaseModel):
    enabled: bool


class SpeakRequest(BaseModel):
    text: str
    language: str | None = None


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def on_startup() -> None:
    LOG.info("=" * 70)
    LOG.info("VISUAL ASSISTANCE SYSTEM - starting")
    LOG.info("=" * 70)

    model_info = STATE.get("model_info") or discover_model()
    en_names, bn_names = load_class_metadata()
    model_info.class_names_bn = bn_names

    # Map English class name -> Bengali, for the message generator.
    bn_overrides = {}
    if en_names and bn_names:
        for cid, en in en_names.items():
            if cid in bn_names:
                bn_overrides[en] = bn_names[cid]
        LOG.info("Loaded Bengali names for %d classes from data.yaml", len(bn_overrides))

    try:
        pipeline = DetectionPipeline(settings, model_info, bn_overrides=bn_overrides)
    except ModelLoadError as exc:
        STATE["startup_error"] = str(exc)
        LOG.error("MODEL LOAD FAILED: %s", exc)
        return
    except Exception as exc:
        STATE["startup_error"] = "Unexpected startup failure: {}".format(exc)
        LOG.exception("Startup failed")
        return

    STATE["pipeline"] = pipeline
    # The source the app started with IS the live camera; a later video
    # upload overwrites camera_source, so remember it now.
    STATE.setdefault("live_camera", settings.get("camera_source"))
    STATE["model_info"] = model_info

    LOG.info("Frontend: %s", FRONTEND_DIR)
    LOG.info("Ready. Open the URL below and press START.")
    LOG.info("%s", DISCLAIMER)


@app.on_event("shutdown")
def on_shutdown() -> None:
    p = STATE.get("pipeline")
    if p:
        p.shutdown()
    LOG.info("Shutdown complete.")


def get_pipeline() -> DetectionPipeline:
    p = STATE.get("pipeline")
    if p is None:
        raise HTTPException(
            status_code=503,
            detail=STATE.get("startup_error") or "Pipeline is not initialised.")
    return p


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    page = FRONTEND_DIR / "index.html"
    if not page.exists():
        raise HTTPException(404, "frontend/index.html is missing")
    return FileResponse(str(page))


@app.get("/style.css")
def style():
    return FileResponse(str(FRONTEND_DIR / "style.css"), media_type="text/css")


@app.get("/app.js")
def appjs():
    return FileResponse(str(FRONTEND_DIR / "app.js"), media_type="application/javascript")


@app.get("/favicon.ico")
def favicon():
    return JSONResponse(status_code=204, content=None)


# --------------------------------------------------------------------------- #
# Video stream
# --------------------------------------------------------------------------- #
def mjpeg_generator(pipeline: DetectionPipeline):
    boundary = b"--frame\r\n"
    last = None
    while True:
        cap = int(settings.get("stream_fps_cap"))
        frame = pipeline.latest_jpeg()
        if frame is None:
            time.sleep(0.05)
            if not pipeline.running:
                # Nothing to stream and nothing coming; let the client retry.
                break
            continue
        if frame is not last:
            last = frame
            yield boundary + b"Content-Type: image/jpeg\r\nContent-Length: " \
                + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
        time.sleep(1.0 / max(cap, 1))


@app.get("/video_feed")
def video_feed():
    pipeline = get_pipeline()
    return StreamingResponse(
        mjpeg_generator(pipeline),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate",
                 "Pragma": "no-cache", "Connection": "close"},
    )


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/api/status")
def api_status():
    p = STATE.get("pipeline")
    if p is None:
        return {"ok": False, "error": STATE.get("startup_error"),
                "disclaimer": DISCLAIMER}
    st = p.status()
    st["ok"] = True
    st["disclaimer"] = DISCLAIMER
    st["settings"] = settings.snapshot()
    return st


@app.get("/api/state")
def api_state():
    """
    One poll for everything the UI needs each tick: detections, guidance and
    any speech the browser must perform. Deliberately a single endpoint so the
    frontend makes ONE request per tick, not five.
    """
    p = STATE.get("pipeline")
    if p is None:
        return {"ok": False, "error": STATE.get("startup_error"),
                "running": False, "detections": [], "guidance": ""}

    guidance = p.latest_guidance()
    return {
        "ok": True,
        "running": p.running,
        "language": settings.get("language"),
        "voice_enabled": settings.get("voice_enabled"),
        "detection_enabled": settings.get("detection_enabled"),
        "fps": round(p.fps, 1),
        "inference_ms": round(p.inference_ms, 1),
        "camera_connected": p.camera.connected if p.camera else False,
        "camera_resolution": (p.camera.status().get("resolution") if p.camera else None),
        "camera_blank": (p.camera.blank_source if p.camera else False),
        "playback": (p.camera.playback() if p.camera else {}),
        "input_mode": settings.get("input_mode"),
        "camera_error": p.camera.last_error if p.camera else None,
        "detections": p.latest_detections(),
        "guidance": guidance.get("text", ""),
        "guidance_reason": guidance.get("reason", ""),
        "guidance_stale": guidance.get("stale", False),
        "urgent": guidance.get("urgent", False),
        "safety": p.navigator.safety_status(p.latest_detections()),
        "speak_browser": p.voice.take_browser_queue(),
        "play_audio": p.voice.take_audio_queue(),
        "model_is_placeholder": p.model_info.is_placeholder,
    }


@app.get("/api/detections")
def api_detections():
    p = get_pipeline()
    g = p.latest_guidance()
    return {
        "running": p.running,
        "language": settings.get("language"),
        "voice_enabled": settings.get("voice_enabled"),
        "detections": p.latest_detections(),
        "guidance": g.get("text", ""),
    }


@app.post("/api/start")
def api_start():
    p = get_pipeline()
    try:
        p.start()
    except CameraError as exc:
        LOG.error("Camera failed to start: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        LOG.exception("Unexpected start failure")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True, "running": p.running, "camera": p.camera.status() if p.camera else None}


@app.post("/api/stop")
def api_stop():
    p = get_pipeline()
    p.stop()
    return {"ok": True, "running": p.running}


@app.get("/api/settings")
def api_get_settings():
    return {"ok": True, "settings": settings.snapshot()}


@app.post("/api/settings")
def api_set_settings(patch: SettingsPatch):
    changes = {k: v for k, v in patch.model_dump().items() if v is not None}
    applied, needs_restart = settings.update(changes)

    p = STATE.get("pipeline")
    if needs_restart and p is not None and p.running:
        try:
            p.restart_camera()
        except CameraError as exc:
            raise HTTPException(status_code=400,
                                detail="Settings saved, but the camera failed to reopen: {}".format(exc))
    if applied and p is not None:
        # New thresholds change what counts as an event; forget stale state.
        p.navigator.reset()
    return {"ok": True, "applied": applied, "camera_restarted": needs_restart,
            "settings": settings.snapshot()}


@app.post("/api/language")
def api_language(patch: LanguagePatch):
    if patch.language not in ("en", "bn"):
        raise HTTPException(400, "language must be 'en' or 'bn'")
    settings.update({"language": patch.language})
    p = STATE.get("pipeline")
    if p:
        p.navigator.reset()
    return {"ok": True, "language": settings.get("language")}


@app.post("/api/voice")
def api_voice(patch: VoicePatch):
    settings.update({"voice_enabled": patch.enabled})
    return {"ok": True, "voice_enabled": settings.get("voice_enabled")}


@app.post("/api/speak")
def api_speak(req: SpeakRequest):
    """Manual speech - used by the frontend's 'Test voice' button."""
    p = get_pipeline()
    lang = req.language or settings.get("language")
    ok = p.voice.speak(req.text, language=lang, urgent=False)
    return {"ok": ok, "backend": p.voice.resolve_backend(lang), "language": lang}


MAX_UPLOAD_BYTES = 200 * 1024 * 1024      # 200 MB

# A single camera frame. Generous enough for a 1280px JPEG, small enough that a
# runaway client cannot post video-sized payloads at 10 fps.
MAX_FRAME_BYTES = 4 * 1024 * 1024         # 4 MB


@app.post("/api/detect_frame")
async def api_detect_frame(file: UploadFile = File(...),
                           language: str | None = Form(default=None),
                           speak: bool = Form(default=True)):
    """
    Detect on ONE frame captured by the client's own camera.

    This is what makes the system usable from a phone. The live path
    (/video_feed) opens a camera on the SERVER with OpenCV, so a phone loading
    that stream sees the laptop's webcam rather than its own. Here the browser
    owns the camera through getUserMedia and posts frames instead, while
    detection, spatial analysis and navigation stay on the server using the
    trained thesis model. Nothing about the model changes.

    Nothing is written to disk: at 5-10 fps an annotated JPEG per call would be
    hundreds of files a minute. Boxes come back normalised and the browser
    draws them over its own video element.

    Speech is returned inline rather than left to the /api/state poll, because
    that endpoint reports the live pipeline and is silent in this mode.
    """
    p = get_pipeline()

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty frame.")
    if len(data) > MAX_FRAME_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Frame is {:.1f} MB; the limit is {:.0f} MB. Lower the "
                   "capture resolution.".format(len(data) / 1e6, MAX_FRAME_BYTES / 1e6))

    # Decode in memory. imdecode returns None for anything that is not a real
    # image, which is what keeps a bad upload away from the model.
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400,
                            detail="Could not decode that frame as an image.")

    lang = language if language in ("en", "bn") else settings.get("language")

    try:
        analyzer = MediaAnalyzer(p, STATIC_DIR / "uploads")
        result = analyzer.analyze_frame(frame, lang, live=True)
    except Exception as exc:
        LOG.exception("Frame detection failed")
        raise HTTPException(status_code=500, detail="Detection failed: {}".format(exc))

    # Hand the phrase to the same voice engine the desktop uses. Whether it
    # comes back as an MP3 URL (edge-tts) or as text for the browser to speak
    # depends on the machine's capabilities, exactly as it does on desktop.
    if speak and result.get("speak") and settings.get("voice_enabled") and result["guidance"]:
        p.voice.speak(result["guidance"], language=lang, urgent=result.get("urgent", False))

    result["play_audio"] = p.voice.take_audio_queue()
    result["speak_browser"] = p.voice.take_browser_queue()
    result["safety"] = p.navigator.safety_status(
        [{"class_name": i["class_name"], "position": i["position"],
          "proximity": i["proximity"], "confidence": i["confidence"]}
         for i in result.get("items", [])]
    )
    result["language"] = lang
    return result


MAX_UPLOAD_BYTES = 200 * 1024 * 1024      # 200 MB


class PlaybackRequest(BaseModel):
    action: str                      # play | pause | restart


@app.post("/api/upload_video")
async def api_upload_video(file: UploadFile = File(...),
                           loop: bool = Form(default=False),
                           autostart: bool = Form(default=True)):
    """
    Store an uploaded clip and make it the pipeline's source.

    The video is then processed by exactly the same loop as the live camera -
    frame in, detection, position, navigation, voice, frame out - so "video
    mode" is a different INPUT, not a second detection implementation. That is
    why pause/restart and the MJPEG view work here without any extra plumbing.
    """
    p = get_pipeline()

    if kind_of(file.filename or "") != "video":
        raise HTTPException(
            status_code=415,
            detail="'{}' is not a video. Use .mp4, .mov, .avi, .mkv or .webm."
                   .format(file.filename))

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded video is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Video is {:.0f} MB; the limit is {:.0f} MB.".format(
                len(data) / 1e6, MAX_UPLOAD_BYTES / 1e6))

    videos = STATIC_DIR / "uploads"
    videos.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "clip.mp4").suffix.lower()[:8] or ".mp4"
    dest = videos / ("vid_" + uuid.uuid4().hex[:12] + suffix)
    dest.write_bytes(data)

    # Remove older uploads so a demo session does not fill the disk. The file
    # just written is kept, obviously.
    try:
        clips = sorted(videos.glob("vid_*"), key=lambda q: q.stat().st_mtime)
        for old in clips[:-3]:
            old.unlink(missing_ok=True)
    except OSError:
        pass

    p.stop()
    settings.update({"camera_source": str(dest), "video_loop": bool(loop),
                     "input_mode": "video"})

    started = False
    if autostart:
        try:
            p.start()
            started = True
        except CameraError as exc:
            raise HTTPException(
                status_code=400,
                detail="Could not open '{}'. The codec may be unsupported: {}"
                       .format(file.filename, exc))

    pb = p.camera.playback() if p.camera else {}
    return {"ok": True, "running": started, "source_name": file.filename,
            "playback": pb}


class ModeRequest(BaseModel):
    mode: str                        # live | image | video
    camera: str | None = None        # source to restore when returning to live


@app.post("/api/mode")
def api_mode(req: ModeRequest):
    """
    Switch input mode.

    Only `live` touches the pipeline: it restores a real camera source, because
    after a video upload `camera_source` points at a file and "Live Camera"
    must mean the camera again. `image` and `video` are handled by their own
    endpoints, so here they only record the selection.
    """
    p = get_pipeline()
    mode = (req.mode or "").strip().lower()
    if mode not in ("live", "image", "video"):
        raise HTTPException(status_code=400,
                            detail="Unknown mode '{}'.".format(req.mode))

    settings.update({"input_mode": mode})

    if mode == "live":
        p.stop()
        target = (req.camera or "").strip() or STATE.get("live_camera") or "0"
        settings.update({"camera_source": target, "video_loop": False})
        return {"ok": True, "mode": mode, "camera_source": target, "running": False}

    return {"ok": True, "mode": mode}


@app.post("/api/playback")
def api_playback(req: PlaybackRequest):
    """play / pause / restart for an uploaded video."""
    p = get_pipeline()
    result = p.playback_command((req.action or "").strip().lower())
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Playback failed."))
    return result


@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...),
                      language: str | None = Form(default=None),
                      speak: bool = Form(default=True)):
    """
    Run detection + navigation over an uploaded image or video.

    Deliberately reuses the live pipeline's detector, spatial analyser and
    navigation engine, so an uploaded file is judged by exactly the same model
    and thresholds as the camera. It does NOT touch the camera, so a live
    session keeps running while a file is analysed.
    """
    p = get_pipeline()

    if kind_of(file.filename or "") == "unsupported":
        raise HTTPException(
            status_code=415,
            detail="Unsupported file '{}'. Upload a .jpg/.png image or an "
                   ".mp4/.mov/.avi video.".format(file.filename))

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File is {:.0f} MB; the limit is {:.0f} MB.".format(
                len(data) / 1e6, MAX_UPLOAD_BYTES / 1e6))

    uploads = STATIC_DIR / "uploads"
    # Keep the extension (OpenCV picks its decoder from it) but never trust the
    # uploaded name as a path - it decides only the suffix.
    suffix = Path(file.filename or "upload").suffix.lower()[:8]
    tmp = uploads / ("in_" + uuid.uuid4().hex[:12] + suffix)
    tmp.write_bytes(data)

    lang = language if language in ("en", "bn") else settings.get("language")

    try:
        analyzer = MediaAnalyzer(p, uploads)
        result = analyzer.analyze(tmp, lang)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        LOG.exception("Analysis failed")
        raise HTTPException(status_code=500, detail="Analysis failed: {}".format(exc))
    finally:
        # The upload itself is not needed once decoded; only the annotated
        # output is served back.
        try:
            tmp.unlink()
        except OSError:
            pass

    # Speak the conclusion, honouring the same voice switch as the live system.
    if speak and settings.get("voice_enabled") and result.get("guidance"):
        p.voice.speak(result["guidance"], language=lang,
                      urgent=bool(result.get("urgent")))

    result["language"] = lang
    result["ok"] = True
    return result


@app.get("/api/cameras")
def api_cameras():
    """Probe device indices so the user never has to guess which one is the phone."""
    found = probe_indices(max_index=4)
    return {"ok": True, "available": found,
            "hint": "Phone-as-webcam apps (DroidCam/Iriun) appear as an extra index, "
                    "usually 1 or 2, only while the app is connected."}


@app.get("/api/model")
def api_model():
    p = get_pipeline()
    info = p.detector.status()
    info["disclaimer"] = DISCLAIMER
    if p.model_info.is_placeholder:
        info["warning"] = (
            "No trained best.pt was found, so the stock COCO model is being used. "
            "Detected classes are NOT your thesis classes. Finish Part 1 and place "
            "best.pt in models/ or runs/train/*/weights/ - it is picked up automatically."
        )
    return info


@app.get("/api/health")
def api_health():
    return {"ok": STATE.get("pipeline") is not None,
            "error": STATE.get("startup_error")}


# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Visual Assistance System - real-time detection and voice guidance.")
    ap.add_argument("--camera", default=None,
                    help="camera index (0/1/2) or stream URL "
                         "(e.g. http://192.168.0.101:8080/video)")
    ap.add_argument("--model", default=None,
                    help="path to trained best.pt (default: auto-discover)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 makes the UI reachable from your phone on the same Wi-Fi")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--confidence", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--language", default=None, choices=["en", "bn"])
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--autostart", action="store_true",
                    help="begin detection immediately instead of waiting for START")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    patch = {}
    if args.camera is not None:
        patch["camera_source"] = args.camera
    if args.confidence is not None:
        patch["confidence"] = args.confidence
    if args.imgsz is not None:
        patch["imgsz"] = args.imgsz
    if args.language is not None:
        patch["language"] = args.language
    if args.device is not None:
        patch["device"] = args.device
    if args.no_voice:
        patch["voice_enabled"] = False
    if patch:
        settings.update(patch)

    STATE["model_info"] = discover_model(args.model)

    import uvicorn
    LOG.info("Starting server on http://%s:%d", args.host, args.port)
    if args.host == "0.0.0.0":
        LOG.info("Reachable from your phone at http://<this-laptop-LAN-IP>:%d", args.port)

    config = uvicorn.Config(app, host=args.host, port=args.port,
                            log_level=args.log_level.lower(), access_log=False)
    server = uvicorn.Server(config)

    if args.autostart:
        import threading

        def _auto():
            # WAIT for the pipeline instead of assuming it exists after a fixed
            # delay. Model download + load can take far longer than any sleep
            # worth hardcoding, and the old version silently did nothing when it
            # lost that race - the app came up stopped with no explanation.
            deadline = time.time() + 120.0
            while time.time() < deadline:
                p = STATE.get("pipeline")
                if p is not None:
                    try:
                        p.start()
                        LOG.info("Autostart: detection started.")
                    except CameraError as exc:
                        LOG.error("Autostart failed to open the camera: %s", exc)
                    except Exception as exc:
                        LOG.exception("Autostart failed: %s", exc)
                    return
                if STATE.get("startup_error"):
                    LOG.error("Autostart aborted - startup failed: %s",
                              STATE["startup_error"])
                    return
                time.sleep(0.25)
            LOG.error("Autostart timed out after 120s waiting for the pipeline.")
        threading.Thread(target=_auto, daemon=True).start()

    try:
        server.run()
    except KeyboardInterrupt:
        LOG.info("Interrupted by user.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
