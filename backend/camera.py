#!/usr/bin/env python3
"""
camera.py
=========
Threaded camera capture.

Why a thread: cv2.VideoCapture.read() blocks. If the inference loop called it
directly, a slow model would build up a backlog of stale frames and the "live"
view would drift seconds behind reality - dangerous for a navigation aid. This
grabber runs flat out and keeps only the NEWEST frame; the pipeline consumes
whatever is current and old frames are dropped.

Sources supported (section 6 / 20):
  * integer index  "0", "1", "2"   - laptop webcam, or a phone exposed as a
                                     webcam by DroidCam / Iriun
  * URL            "http://192.168.0.101:8080/video"  - IP Webcam style MJPEG
                   "rtsp://..."                       - RTSP stream
No mobile app is hardcoded; anything OpenCV can open works.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2

LOG = logging.getLogger("camera")


class CameraError(RuntimeError):
    """Raised when a source cannot be opened at all."""


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")


def parse_source(raw: str):
    """'0' -> 0 (device index); anything else stays a string (URL/path)."""
    raw = str(raw).strip()
    if raw.isdigit():
        return int(raw)
    return raw


def is_video_file(raw: str) -> bool:
    """
    A local video file, as opposed to a device index or a network stream.

    Files need two behaviours a live camera does not: play at the file's own
    frame rate rather than as fast as the disk allows, and rewind at the end
    instead of being treated as a disconnected camera. This makes a recorded
    clip a reproducible stand-in for the camera - useful for evaluation runs
    and for demonstrating the system where no camera is available.
    """
    raw = str(raw).strip().lower()
    if raw.isdigit() or "://" in raw:
        return False
    return raw.endswith(VIDEO_EXTS)


# --------------------------------------------------------------------------- #
# Blank-frame detection
# --------------------------------------------------------------------------- #
# A camera that opens successfully and returns frames is NOT necessarily a
# working camera. Virtual cameras (NVIDIA Broadcast, Camo, OBS, DroidCam) stay
# open and stream a solid black or placeholder image when no real source is
# attached, and some backends return black for a device that works fine on
# another backend.
#
# Without this check the application streams a perfectly black feed, reports a
# healthy FPS, detects nothing, and looks like a broken MODEL rather than a
# disconnected CAMERA. That is a genuinely expensive thing to debug, so the
# capture layer validates its own frames and says so.

BLANK_MEAN = 6.0     # near-black
BLANK_STD = 6.0      # and essentially no variation => no real image
WARMUP_FRAMES = 10   # auto-exposure needs a moment; judging frame 1 is unfair


def frame_is_blank(frame) -> bool:
    """True when a frame carries no usable image (solid black / flat colour)."""
    if frame is None or frame.size == 0:
        return True
    return float(frame.mean()) < BLANK_MEAN and float(frame.std()) < BLANK_STD


def _warm_and_grab(cap, frames: int = WARMUP_FRAMES):
    """Read a few frames and return the most informative one."""
    best, best_std = None, -1.0
    for _ in range(max(1, frames)):
        ok, f = cap.read()
        if ok and f is not None:
            std = float(f.std())
            if std > best_std:
                best, best_std = f, std
    return best


class CameraSource:
    """
    Background frame grabber with automatic reconnection.

    Usage:
        cam = CameraSource("0", 960, 540)
        cam.open()                 # raises CameraError on failure
        ok, frame = cam.read()     # newest frame, or (False, None)
        cam.close()
    """

    def __init__(self, source: str = "0", width: int = 960, height: int = 540,
                 reconnect_delay: float = 2.0, loop: bool = True):
        self.source = source
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay

        self._cap: cv2.VideoCapture | None = None
        self._frame = None
        self._seq = 0                      # increments per new frame
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self.connected = False
        self.last_error: str | None = None
        self.capture_fps = 0.0
        self.frames_read = 0
        self.actual_width = 0
        self.actual_height = 0

        self.is_file = is_video_file(source)
        self.blank_source = False
        self.loops = 0                 # times a video file has rewound
        self._file_frame_interval = 0.0

        # ---- playback control (video-file sources only) ----
        # A live camera cannot be paused or rewound - the world does not wait.
        # A recorded file can, and a thesis demo needs exactly that, so these
        # are honoured only when self.is_file is True.
        self.loop = loop
        self.paused = False
        self.finished = False          # reached the end with loop disabled
        self.frame_index = 0           # frames decoded since the last restart
        self.total_frames = 0
        self.duration_s = 0.0
        self._seek_to = None            # set by restart(), applied in the loop

    # ------------------------------------------------------------------ #
    def _build_capture(self) -> cv2.VideoCapture | None:
        src = parse_source(self.source)

        # CAP_DSHOW is markedly more reliable than MSMF for USB/virtual webcams
        # on Windows (which is how DroidCam and Iriun present themselves).
        # MSMF is listed alongside DSHOW because on some Windows machines a
        # device streams pure black on one backend while working on the other.
        # Trying only DSHOW (as this did originally) meant the app could open a
        # camera, report a healthy FPS, and show nothing but black - with a
        # working backend sitting right there unused.
        attempts = []
        if isinstance(src, int):
            attempts = [(src, cv2.CAP_DSHOW), (src, cv2.CAP_MSMF), (src, cv2.CAP_ANY)]
        else:
            attempts = [(src, cv2.CAP_FFMPEG), (src, cv2.CAP_ANY)]

        # Try every backend, and PREFER one that returns a real image. A
        # backend that only yields black is kept as a last resort so the app
        # still starts (and warns) rather than refusing outright.
        fallback = None
        self.blank_source = False

        for target, backend in attempts:
            try:
                cap = cv2.VideoCapture(target, backend)
            except Exception as exc:
                LOG.debug("VideoCapture(%r, %s) raised: %s", target, backend, exc)
                continue
            if not cap.isOpened():
                cap.release()
                continue

            if isinstance(src, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # Keep the driver buffer tiny so read() returns fresh frames.
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # A video file is not judged on brightness - a clip may legitimately
            # open on a dark frame.
            if self.is_file:
                return cap

            probe = _warm_and_grab(cap)
            if probe is None:
                cap.release()
                continue
            if not frame_is_blank(probe):
                LOG.info("Camera %r opened on backend %s", target, backend)
                return cap

            LOG.warning("Camera %r on backend %s returns blank frames - trying another",
                        target, backend)
            if fallback is None:
                fallback = cap
            else:
                cap.release()

        if fallback is not None:
            self.blank_source = True
            LOG.warning("=" * 70)
            LOG.warning("CAMERA OPENED BUT IS DELIVERING A BLANK (BLACK) IMAGE")
            LOG.warning("The detector cannot find anything in a black frame - this is")
            LOG.warning("a camera problem, not a model problem. Usually one of:")
            LOG.warning("  * a virtual camera (Camo / NVIDIA Broadcast / OBS / DroidCam)")
            LOG.warning("    is selected but no phone or source is connected to it")
            LOG.warning("  * a privacy shutter is closed, or the camera is disabled")
            LOG.warning("  * another application is already holding the camera")
            LOG.warning("Run: python -m backend.camera   to list working sources.")
            LOG.warning("=" * 70)
            return fallback
        return None

    # ------------------------------------------------------------------ #
    def open(self) -> None:
        """Open the source and start the grabber thread."""
        self.close()
        self._stop.clear()

        cap = self._build_capture()
        if cap is None:
            self.connected = False
            self.last_error = self._diagnose()
            raise CameraError(self.last_error)

        # Confirm we can actually pull a frame - some virtual cameras open
        # successfully but never deliver anything.
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            self.connected = False
            self.last_error = (
                "Camera source '{}' opened but delivered no frames. If this is a "
                "phone webcam app, make sure the app is running and connected."
                .format(self.source))
            raise CameraError(self.last_error)

        self._cap = cap
        self.actual_height, self.actual_width = frame.shape[:2]

        if self.is_file:
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            declared = cap.get(cv2.CAP_PROP_FPS) or 0.0
            # Guard against nonsense metadata (0, NaN, absurd values).
            if not (1.0 <= declared <= 120.0):
                declared = 25.0
            self._file_frame_interval = 1.0 / declared
            self.duration_s = (self.total_frames / declared) if self.total_frames else 0.0
            self.frame_index = 0
            self.finished = False
            LOG.info("Video file: %.1f fps, %d frames, %.1fs, loop=%s",
                     declared, self.total_frames, self.duration_s, self.loop)
        with self._lock:
            self._frame = frame
            self._seq += 1
        self.connected = True
        self.last_error = None
        self.frames_read = 1

        self._thread = threading.Thread(target=self._grab_loop, name="camera-grab", daemon=True)
        self._thread.start()
        LOG.info("Camera opened: source=%s actual=%dx%d",
                 self.source, self.actual_width, self.actual_height)

    # ------------------------------------------------------------------ #
    def _diagnose(self) -> str:
        src = parse_source(self.source)
        if isinstance(src, int):
            available = probe_indices(max_index=5)
            hint = ("Detected working camera indices: {}".format(available)
                    if available else
                    "No working camera index was detected at all.")
            return (
                "Could not open camera index {}. {} "
                "If you are using a phone as a webcam, start DroidCam/Iriun on BOTH "
                "phone and PC first, then retry - the virtual camera only appears "
                "once the client is connected.".format(src, hint))
        return (
            "Could not open stream URL '{}'. Check that the phone and laptop are on "
            "the same Wi-Fi, that the URL is exactly what the app shows (IP Webcam "
            "usually needs the '/video' suffix), and that no firewall is blocking it."
            .format(self.source))

    # ------------------------------------------------------------------ #
    def _grab_loop(self) -> None:
        last_t = time.perf_counter()
        fails = 0
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                break
            try:
                ok, frame = cap.read()
            except Exception as exc:
                ok, frame = False, None
                LOG.debug("read() raised: %s", exc)

            # ---- playback control, before any decoding ---- #
            if self.is_file:
                if self._seek_to is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(self._seek_to))
                    self.frame_index = int(self._seek_to)
                    self._seek_to = None
                    self.finished = False
                if self.paused or self.finished:
                    # Hold the last frame on screen rather than going black.
                    time.sleep(0.05)
                    continue

            if (not ok or frame is None) and self.is_file:
                # End of file. Rewind when looping; otherwise stop and say so.
                if not self.loop:
                    self.finished = True
                    LOG.info("Video finished: %s", self.source)
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_index = 0
                self.loops += 1
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.connected = False
                    self.last_error = "Video file '{}' could not be re-read.".format(self.source)
                    break

            if not ok or frame is None:
                fails += 1
                if fails >= 15:
                    LOG.warning("Camera '%s' stopped delivering frames - reconnecting", self.source)
                    self.connected = False
                    self._reconnect()
                    fails = 0
                else:
                    time.sleep(0.01)
                continue

            fails = 0
            self.frames_read += 1
            if self.is_file:
                self.frame_index += 1
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 0:
                # exponential moving average keeps the number readable
                self.capture_fps = (0.9 * self.capture_fps + 0.1 * (1.0 / dt)) if self.capture_fps else 1.0 / dt

            with self._lock:
                self._frame = frame
                self._seq += 1

            if self._file_frame_interval:
                # Play in real time; without this a file is consumed as fast as
                # the disk allows and the "FPS" reading becomes meaningless.
                time.sleep(self._file_frame_interval)
                self.actual_height, self.actual_width = frame.shape[:2]
            self.connected = True

    # ------------------------------------------------------------------ #
    def _reconnect(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        while not self._stop.is_set():
            time.sleep(self.reconnect_delay)
            cap = self._build_capture()
            if cap is not None:
                self._cap = cap
                self.connected = True
                self.last_error = None
                LOG.info("Camera '%s' reconnected", self.source)
                return
            self.last_error = "Lost connection to '{}'; retrying...".format(self.source)
            LOG.warning(self.last_error)

    # ------------------------------------------------------------------ #
    def read(self):
        """Newest frame. Returns (ok, frame). Frame is NOT copied - do not mutate."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame

    def read_new(self, last_seq: int):
        """Newest frame plus its sequence number, for skipping duplicates."""
        with self._lock:
            if self._frame is None:
                return False, None, last_seq
            return True, self._frame, self._seq

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self.connected = False

    # ------------------------------------------------------------------ #
    # Playback control. No-ops on a live camera by design: pausing reality is
    # not a thing, and silently pretending to would mislead the caller.
    # ------------------------------------------------------------------ #
    def pause(self) -> bool:
        if not self.is_file:
            return False
        self.paused = True
        return True

    def resume(self) -> bool:
        if not self.is_file:
            return False
        # Resuming after the clip ended replays it from the start, which is
        # what "play" means to someone looking at a finished video.
        if self.finished:
            self._seek_to = 0
        self.paused = False
        return True

    def restart(self) -> bool:
        if not self.is_file:
            return False
        self._seek_to = 0
        self.finished = False
        self.paused = False
        return True

    def playback(self) -> dict:
        """Progress for the UI. Empty dict for a live camera - it has no length."""
        if not self.is_file:
            return {}
        fps = (1.0 / self._file_frame_interval) if self._file_frame_interval else 0.0
        pos = (self.frame_index / fps) if fps else 0.0
        return {
            "is_file": True,
            "paused": self.paused,
            "finished": self.finished,
            "loop": self.loop,
            "frame": self.frame_index,
            "total_frames": self.total_frames,
            "position_s": round(pos, 2),
            "duration_s": round(self.duration_s, 2),
            "percent": round(100.0 * self.frame_index / self.total_frames, 1)
                       if self.total_frames else 0.0,
            "loops": self.loops,
        }

    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        return {
            "source": self.source,
            "connected": self.connected,
            "resolution": "{}x{}".format(self.actual_width, self.actual_height)
            if self.actual_width else None,
            "capture_fps": round(self.capture_fps, 1),
            "frames_read": self.frames_read,
            "error": self.last_error,
            "blank_source": self.blank_source,
            "playback": self.playback(),
        }


# --------------------------------------------------------------------------- #
def probe_indices(max_index: int = 5) -> list[int]:
    """
    Camera indices that deliver a REAL image, not just a stream of black.

    Reports only usable sources, because an index that opens and returns black
    is worse than one that fails outright: it looks like it works.
    Opening cameras is slow, so keep max_index small.
    """
    return [c["index"] for c in probe_detailed(max_index) if not c["blank"]]


def probe_detailed(max_index: int = 5) -> list[dict]:
    """
    Full report per index: which backend worked and whether the image is blank.
    Used by the UI's camera picker and by the `python -m backend.camera` helper.
    """
    out: list[dict] = []
    for i in range(max_index + 1):
        for name, backend in (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)):
            cap = None
            try:
                cap = cv2.VideoCapture(i, backend)
                if not cap.isOpened():
                    continue
                frame = _warm_and_grab(cap)
                if frame is None:
                    continue
                blank = frame_is_blank(frame)
                existing = next((c for c in out if c["index"] == i), None)
                info = {
                    "index": i,
                    "backend": name,
                    "resolution": "{}x{}".format(frame.shape[1], frame.shape[0]),
                    "blank": blank,
                }
                # A working backend replaces a blank one for the same index.
                if existing is None:
                    out.append(info)
                elif existing["blank"] and not blank:
                    out[out.index(existing)] = info
            except Exception:
                pass
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
    return out


if __name__ == "__main__":
    # python -m backend.camera  -> tell the user what their machine can see.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=" * 70)
    print("CAMERA SOURCES ON THIS MACHINE")
    print("=" * 70)
    cams = probe_detailed(5)
    if not cams:
        print("  No camera could be opened at all.")
    for c in cams:
        state = "BLANK / no signal" if c["blank"] else "OK - real image"
        print("  index {}  [{}]  {:<10}  {}".format(
            c["index"], c["backend"], c["resolution"], state))
    usable = [c for c in cams if not c["blank"]]
    print()
    if usable:
        print("  Use:  python app/main.py --camera {}".format(usable[0]["index"]))
    else:
        print("  No source is producing a real image.")
        print()
        print("  Every camera here is a VIRTUAL camera with nothing attached.")
        print("  Options:")
        print("    * Camo:        start Camo on the phone AND Camo Studio on the PC")
        print("    * IP Webcam:   install an IP-webcam app, then")
        print("                   python app/main.py --camera http://<phone-ip>:8080/video")
        print("    * Video file:  python app/main.py --camera clip.mp4")
    print("=" * 70)
