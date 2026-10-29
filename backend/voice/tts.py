#!/usr/bin/env python3
"""
tts.py
======
Non-blocking text-to-speech with pluggable backends (sections 13, 29, 30).

THE BENGALI PROBLEM - measured on this machine, not assumed
-----------------------------------------------------------
`pyttsx3` drives Windows SAPI5 and can only use the voices Windows has
installed. This laptop has exactly two:
    Microsoft David Desktop  - en-US
    Microsoft Zira Desktop   - en-US
There is NO Bengali voice, so pyttsx3 physically cannot speak Bengali here.

Backends, and what each actually gives you:

  pyttsx3   offline, zero latency, ENGLISH ONLY on this machine.
            To get offline Bengali you must install a Bengali SAPI voice:
            Windows Settings -> Time & language -> Language & region
            -> Add a language -> "বাংলা (বাংলাদেশ)" -> tick "Speech".
            Then it appears here automatically and this backend handles bn too.

  server    edge-tts (Microsoft Edge neural voices) synthesises MP3 on the
            backend, the browser plays it. REQUIRES INTERNET.
            Verified available Bengali voices:
                bn-BD-NabanitaNeural (female, Bangladesh)
                bn-BD-PradeepNeural  (male,   Bangladesh)
                bn-IN-BashkarNeural  (male,   India)
                bn-IN-TanishaaNeural (female, India)
            Free, no API key. Best Bengali quality available. DEFAULT for bn.

  browser   the frontend speaks via the Web Speech API. Offline once the voice
            is on the device, and audio comes out of the PHONE when the phone
            is the client - which is what a real user would want. Bengali
            support depends on the browser/OS: Android Chrome usually has
            bn-BD; desktop Chrome on Windows often does not.

  off       silent; text still appears in the UI.

`auto` picks: English -> pyttsx3 (instant, offline); Bengali -> server if
edge-tts imports, else browser, else off with a clear warning.

THREADING
---------
speak() only enqueues. A daemon worker thread does synthesis and playback, so
the camera and inference loops never block on audio. The queue is bounded and
drops the OLDEST pending phrase when full - for a navigation aid, a stale
warning is worse than no warning.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("tts")

AUDIO_CACHE = Path(__file__).resolve().parent.parent.parent / "static" / "audio"

BENGALI_VOICES = [
    "bn-BD-NabanitaNeural",
    "bn-BD-PradeepNeural",
    "bn-IN-BashkarNeural",
    "bn-IN-TanishaaNeural",
]
ENGLISH_VOICE = "en-US-AriaNeural"


@dataclass
class Utterance:
    text: str
    language: str
    urgent: bool = False
    id: str = ""


# --------------------------------------------------------------------------- #
def probe_backends() -> dict:
    """What this machine can genuinely do. Called once at startup and exposed
    through /api/status so the UI can tell the truth about Bengali."""
    caps = {
        "pyttsx3": False,
        "pyttsx3_voices": [],
        "pyttsx3_has_bengali": False,
        "edge_tts": False,
        "gtts": False,
        "notes": [],
    }
    try:
        import pyttsx3
        eng = pyttsx3.init()
        voices = eng.getProperty("voices") or []
        caps["pyttsx3"] = True
        for v in voices:
            langs = []
            for L in getattr(v, "languages", []) or []:
                langs.append(L.decode(errors="ignore") if isinstance(L, bytes) else str(L))
            caps["pyttsx3_voices"].append({"name": v.name, "languages": langs})
            blob = (v.name + " " + " ".join(langs)).lower()
            if "bn" in blob.split() or "bengali" in blob or "bangla" in blob:
                caps["pyttsx3_has_bengali"] = True
        try:
            eng.stop()
        except Exception:
            pass
    except Exception as exc:
        caps["notes"].append("pyttsx3 unavailable: {}".format(exc))

    try:
        import edge_tts  # noqa: F401
        caps["edge_tts"] = True
    except Exception:
        caps["notes"].append("edge-tts not installed (pip install edge-tts) - "
                             "needed for high-quality Bengali speech")
    try:
        import gtts  # noqa: F401
        caps["gtts"] = True
    except Exception:
        pass

    if not caps["pyttsx3_has_bengali"]:
        caps["notes"].append(
            "No Bengali voice is installed in Windows, so OFFLINE Bengali speech "
            "is not possible. Bengali will use edge-tts (needs internet) or the "
            "browser's own voice.")
    return caps


# --------------------------------------------------------------------------- #
class VoiceEngine:
    """
    Thread-safe, non-blocking speech.

        voice = VoiceEngine(settings)
        voice.start()
        voice.speak("There is a pole ahead.", language="en")
        voice.stop()
    """

    MAX_QUEUE = 3

    def __init__(self, settings):
        self.settings = settings
        self.caps = probe_backends()

        self._queue: queue.Queue[Utterance] = queue.Queue(maxsize=self.MAX_QUEUE)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._pyttsx3 = None
        self._pyttsx3_failed = False

        # Phrases the BROWSER should speak, polled by the frontend.
        self._browser_queue: list[dict] = []
        # Audio files the browser should play (server backend).
        self._pending_audio: list[dict] = []

        self.spoken_count = 0
        self.dropped_count = 0
        self.last_spoken = ""
        self.last_error: str | None = None
        self.active_backend = "unknown"

        AUDIO_CACHE.mkdir(parents=True, exist_ok=True)
        self._cleanup_cache()

    # ------------------------------------------------------------------ #
    def _cleanup_cache(self, keep_seconds: float = 600) -> None:
        """Audio clips are ephemeral; do not let them accumulate."""
        now = time.time()
        try:
            for f in AUDIO_CACHE.glob("*.mp3"):
                if now - f.stat().st_mtime > keep_seconds:
                    f.unlink(missing_ok=True)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def resolve_backend(self, language: str) -> str:
        """Which backend will actually be used for this language right now."""
        pref = self.settings.get("tts_backend")
        if pref == "off":
            return "off"
        if pref != "auto":
            # Honour the explicit choice, but refuse an impossible combination.
            if pref == "pyttsx3" and language == "bn" and not self.caps["pyttsx3_has_bengali"]:
                LOG.warning("pyttsx3 cannot speak Bengali on this machine; using browser")
                return "browser"
            return pref
        if language == "bn":
            if self.caps["edge_tts"]:
                return "server"
            return "browser"
        if self.caps["pyttsx3"]:
            return "pyttsx3"
        if self.caps["edge_tts"]:
            return "server"
        return "browser"

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="tts-worker", daemon=True)
        self._thread.start()
        LOG.info("Voice engine initialized. pyttsx3=%s edge-tts=%s bengali-offline=%s",
                 self.caps["pyttsx3"], self.caps["edge_tts"], self.caps["pyttsx3_has_bengali"])

    def stop(self) -> None:
        """Stop the worker and silence anything currently playing."""
        self._stop.set()
        self._drain()
        if self._pyttsx3 is not None:
            try:
                self._pyttsx3.stop()
            except Exception:
                pass
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                return

    # ------------------------------------------------------------------ #
    def speak(self, text: str, language: str = "en", urgent: bool = False) -> bool:
        """Enqueue a phrase. Returns False if it was dropped. Never blocks."""
        if not text or not self.settings.get("voice_enabled"):
            return False
        if self._stop.is_set():
            return False

        utt = Utterance(text=text, language=language, urgent=urgent, id=uuid.uuid4().hex[:12])

        # An urgent phrase jumps the queue: clear stale items first.
        if urgent:
            self._drain()

        try:
            self._queue.put_nowait(utt)
            return True
        except queue.Full:
            # Drop the OLDEST, keep the newest - stale guidance is worse than none.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._queue.put_nowait(utt)
                self.dropped_count += 1
                return True
            except (queue.Empty, queue.Full):
                self.dropped_count += 1
                return False

    # ------------------------------------------------------------------ #
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                utt = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._dispatch(utt)
                self.spoken_count += 1
                self.last_spoken = utt.text
            except Exception as exc:
                self.last_error = str(exc)
                LOG.error("TTS failed: %s", exc)
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------ #
    def _dispatch(self, utt: Utterance) -> None:
        backend = self.resolve_backend(utt.language)
        self.active_backend = backend

        if backend == "off":
            return
        if backend == "browser":
            self._enqueue_browser(utt)
            return
        if backend == "pyttsx3":
            if self._speak_pyttsx3(utt):
                return
            LOG.warning("pyttsx3 failed; falling back to browser speech")
            self._enqueue_browser(utt)
            return
        if backend == "server":
            if self._speak_server(utt):
                return
            LOG.warning("server TTS failed; falling back to browser speech")
            self._enqueue_browser(utt)

    # ------------------------------------------------------------------ #
    def _enqueue_browser(self, utt: Utterance) -> None:
        with self._lock:
            self._browser_queue.append({
                "id": utt.id, "text": utt.text,
                "lang": "bn-BD" if utt.language == "bn" else "en-US",
                "ts": time.time(),
            })
            self._browser_queue = self._browser_queue[-3:]

    # ------------------------------------------------------------------ #
    def _speak_pyttsx3(self, utt: Utterance) -> bool:
        if self._pyttsx3_failed or not self.caps["pyttsx3"]:
            return False
        try:
            import pyttsx3
            # A fresh engine per utterance is deliberate: reusing a SAPI5 engine
            # across threads on Windows deadlocks after a few calls.
            eng = pyttsx3.init()
            eng.setProperty("rate", self.settings.get("tts_rate"))
            eng.say(utt.text)
            eng.runAndWait()
            try:
                eng.stop()
            except Exception:
                pass
            return True
        except Exception as exc:
            LOG.error("pyttsx3 error: %s", exc)
            self.last_error = str(exc)
            return False

    # ------------------------------------------------------------------ #
    def _speak_server(self, utt: Utterance) -> bool:
        """Synthesise to MP3 with edge-tts; the browser plays the file."""
        if not self.caps["edge_tts"]:
            return False
        try:
            import asyncio
            import edge_tts

            voice = (self.settings.get("tts_voice_bn")
                     if utt.language == "bn" else ENGLISH_VOICE)
            if utt.language == "bn" and voice not in BENGALI_VOICES:
                voice = BENGALI_VOICES[0]

            out = AUDIO_CACHE / "{}.mp3".format(utt.id)

            async def _synth():
                await edge_tts.Communicate(utt.text, voice).save(str(out))

            # Each call gets its own loop; we are on a dedicated worker thread.
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_synth())
            finally:
                loop.close()

            if not out.exists() or out.stat().st_size == 0:
                return False

            with self._lock:
                self._pending_audio.append({
                    "id": utt.id, "url": "/static/audio/{}.mp3".format(utt.id),
                    "text": utt.text, "lang": utt.language, "ts": time.time(),
                })
                self._pending_audio = self._pending_audio[-3:]
            self._cleanup_cache()
            return True
        except Exception as exc:
            LOG.error("edge-tts error: %s", exc)
            self.last_error = str(exc)
            return False

    # ------------------------------------------------------------------ #
    def take_browser_queue(self) -> list[dict]:
        """Phrases for the frontend to speak; returned once then cleared."""
        with self._lock:
            items, self._browser_queue = self._browser_queue, []
        return items

    def take_audio_queue(self) -> list[dict]:
        """Server-synthesised clips for the frontend to play; returned once."""
        with self._lock:
            items, self._pending_audio = self._pending_audio, []
        return items

    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        lang = self.settings.get("language")
        return {
            "enabled": self.settings.get("voice_enabled"),
            "backend_setting": self.settings.get("tts_backend"),
            "backend_active": self.resolve_backend(lang),
            "language": lang,
            "spoken": self.spoken_count,
            "dropped": self.dropped_count,
            "last_spoken": self.last_spoken,
            "error": self.last_error,
            "capabilities": self.caps,
            "bengali_offline_possible": self.caps["pyttsx3_has_bengali"],
            "bengali_voices": BENGALI_VOICES if self.caps["edge_tts"] else [],
        }
