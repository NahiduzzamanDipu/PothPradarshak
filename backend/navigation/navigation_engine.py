#!/usr/bin/env python3
"""
navigation_engine.py
====================
Decides WHAT to say and, just as importantly, WHEN TO STAY QUIET.

This is the module that separates a demo from something a blind person could
actually tolerate. A detector at 15 FPS produces ~900 detections a minute; a
system that narrates all of them is useless. Three mechanisms control that:

1. PRIORITY (section 12)
   Every detection gets a score. Danger class, CENTER position and NEAR
   proximity all raise it. Only the top item(s) are ever considered.

2. EVENT DETECTION (section 15)
   Speech is triggered by CHANGES, not by presence. A pole that has been on
   your left for 20 seconds is not news. These are events:
       - a new important object appears
       - proximity worsens   (FAR -> MEDIUM -> NEAR)
       - position moves into CENTER from LEFT/RIGHT
       - a danger-class object enters CENTER

3. COOLDOWN (section 14)
   Per-object-and-position cooldown, plus a global minimum gap between any two
   utterances. Both configurable.

Tracking key is (class_name, position). Deliberately NOT the box identity: we
have no tracker, and for guidance "a person on your left" is the same news
whether or not it is literally the same person as last frame.
"""

from __future__ import annotations

import logging
import time

from voice.message_generator import generate_multi_message, clear_path_message

LOG = logging.getLogger("navigation")

# Ranks for detecting a worsening trend.
PROXIMITY_RANK = {"FAR": 0, "MEDIUM": 1, "NEAR": 2}

# Classes that justify an urgent tone and bypass the "announce_far" filter.
# Falls / open holes cause injury; a distant car does not.
DANGER_CLASSES = {
    # thesis classes
    "open_drain", "pothole", "stairs", "curb", "pole",
    # COCO classes that matter for a pedestrian
    "car", "motorcycle", "bus", "truck", "train", "bicycle",
}

# Objects that are never worth announcing even if detected. Keeps the
# placeholder COCO model from narrating furniture and food.
IGNORE_CLASSES = {
    "tie", "frisbee", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "remote", "keyboard", "mouse", "cell phone", "toaster",
}

# Indoor furniture contributed by the secondary model. These are things a
# person can walk into, so they are announced - but they are collision
# hazards, not fall hazards, so they rank BELOW a pothole or an open drain
# and do not get the urgent tone reserved for those.
#
# `oven` and `microwave` were removed from IGNORE_CLASSES above: they were
# listed there to stop the old COCO placeholder narrating a kitchen, but they
# are now deliberately-selected indoor obstacles and must not be silently
# dropped by a filter that contradicts the detector.
INDOOR_OBSTACLES = {
    "chair", "couch", "bed", "dining table", "tv", "toilet",
    "sink", "refrigerator", "oven", "microwave", "potted plant", "bench",
}


class NavigationEngine:
    """
        engine = NavigationEngine(settings, bn_overrides={...})
        result = engine.generate_guidance(detections, language="en")
        # -> {"text": ..., "speak": bool, "urgent": bool, "items": [...]}
    """

    # How long a sentence may remain on screen after its object disappeared.
    # Short on purpose: stale guidance that contradicts the detection table is
    # worse than a brief "path is clear".
    STALE_GUIDANCE_S = 1.0

    def __init__(self, settings, bn_overrides: dict | None = None):
        self.settings = settings
        self.bn_overrides = bn_overrides or {}

        # key -> {"proximity", "position", "last_seen", "last_announced"}
        self._tracked: dict[tuple, dict] = {}
        self._last_announcement_t = 0.0
        self._last_text = ""
        self._current_guidance = ""
        self._guidance_t = 0.0

        self.announcements = 0
        self.suppressed = 0

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear memory - call when the camera restarts, so the first frame of a
        new scene is treated as new rather than as a continuation."""
        self._tracked.clear()
        self._last_announcement_t = 0.0
        self._last_text = ""
        self._current_guidance = ""

    # ------------------------------------------------------------------ #
    def priority_score(self, det: dict) -> float:
        """
        Higher = more urgent. Composed so the ordering is explainable in a viva
        rather than being a tuned magic number.
        """
        score = 0.0
        if det["class_name"] in DANGER_CLASSES:
            score += 100.0
        elif det["class_name"] in INDOOR_OBSTACLES:
            # Worth announcing, but a chair is never more urgent than a hole.
            score += 40.0
        if det.get("position") == "CENTER":
            score += 50.0
        prox = det.get("proximity", "FAR")
        score += {"NEAR": 30.0, "MEDIUM": 12.0, "FAR": 0.0}.get(prox, 0.0)
        # Confidence breaks ties without ever outweighing geometry.
        score += float(det.get("confidence", 0.0)) * 10.0
        return score

    # ------------------------------------------------------------------ #
    def _is_urgent(self, det: dict) -> bool:
        return (det.get("position") == "CENTER"
                and det.get("proximity") == "NEAR")

    # ------------------------------------------------------------------ #
    def _relevant(self, detections: list[dict]) -> list[dict]:
        """Filter by confidence, ignore-list and the announce_far preference."""
        s = self.settings
        conf_min = s.get("confidence")
        announce_far = s.get("announce_far")

        out = []
        for d in detections:
            if d.get("confidence", 0) < conf_min:
                continue
            if d["class_name"] in IGNORE_CLASSES:
                continue
            if (not announce_far
                    and d.get("proximity") == "FAR"
                    and d["class_name"] not in DANGER_CLASSES):
                continue
            out.append(d)
        return out

    # ------------------------------------------------------------------ #
    def _classify_event(self, key, det: dict, now: float) -> str | None:
        """
        Return the reason this detection is newsworthy, or None if it is not.

        CRITICAL: comparison is against the state at the LAST ANNOUNCEMENT, not
        the last frame. If it compared against the last frame, an escalation
        that happened to be suppressed by the cooldown would be consumed and
        silently lost - and the escalation most likely to be suppressed is
        MEDIUM->NEAR, the one the user most needs to hear. Comparing against
        announced state keeps the event PENDING until it is actually spoken.
        """
        prev = self._tracked.get(key)
        prox = det.get("proximity", "FAR")
        pos = det.get("position", "CENTER")

        if prev is None:
            return "new_object"

        # Re-appearance after an absence counts as new.
        if now - prev["last_seen"] > 3.0:
            return "reappeared"

        # Never announced yet (e.g. first sighting was gap-suppressed).
        ann_prox = prev.get("announced_proximity")
        ann_pos = prev.get("announced_position")
        if ann_prox is None:
            return "pending_first_announcement"

        if PROXIMITY_RANK.get(prox, 0) > PROXIMITY_RANK.get(ann_prox, 0):
            return "closer:{}->{}".format(ann_prox, prox)

        if pos == "CENTER" and ann_pos != "CENTER":
            return "moved_into_path:{}->CENTER".format(ann_pos)

        return None

    # ------------------------------------------------------------------ #
    def generate_guidance(self, detections: list[dict], language: str | None = None) -> dict:
        """
        Main entry point. Called once per processed frame.

        Returns:
            {
              "text":    sentence to display (always populated when objects exist)
              "speak":   whether it should be SPOKEN this frame
              "urgent":  whether it warrants an alert tone
              "reason":  why it was announced (or why not)
              "items":   the prioritised objects the text describes
            }
        """
        now = time.time()
        s = self.settings
        language = language or s.get("language")

        relevant = self._relevant(detections)

        if not relevant:
            # Expire stale tracks so a re-entering object counts as new.
            self._tracked = {k: v for k, v in self._tracked.items()
                             if now - v["last_seen"] < 5.0}

            # Hold the last sentence only BRIEFLY. Holding it for seconds makes
            # the UI claim an object is present that the detection table no
            # longer lists - actively misleading in a navigation aid. After the
            # grace period, say the path is clear instead.
            age = now - self._guidance_t
            if self._current_guidance and age <= self.STALE_GUIDANCE_S:
                return {
                    "text": self._current_guidance, "speak": False, "urgent": False,
                    "reason": "no_relevant_objects(holding {:.1f}s)".format(age),
                    "stale": True, "items": [],
                }
            self._current_guidance = ""
            return {
                "text": clear_path_message(language),
                "speak": False, "urgent": False,
                "reason": "no_relevant_objects", "stale": False, "items": [],
            }

        ranked = sorted(relevant, key=self.priority_score, reverse=True)

        # ---- find the highest-priority object that constitutes an EVENT ---- #
        triggering, reason = None, None
        for det in ranked:
            key = (det["class_name"], det.get("position", "CENTER"))
            ev = self._classify_event(key, det, now)
            if ev and triggering is None:
                triggering, reason = det, ev
            # Record state for every relevant object, event or not.
            prior = self._tracked.get(key, {})
            self._tracked[key] = {
                "proximity": det.get("proximity", "FAR"),
                "position": det.get("position", "CENTER"),
                "last_seen": now,
                "last_announced": prior.get("last_announced", 0.0),
                # None until this object has actually been spoken about.
                "announced_proximity": prior.get("announced_proximity"),
                "announced_position": prior.get("announced_position"),
            }

        # Prune tracks not seen recently.
        self._tracked = {k: v for k, v in self._tracked.items()
                         if now - v["last_seen"] < 5.0}

        top = ranked[0]
        urgent = self._is_urgent(top)

        # ---- build the sentence (always, for on-screen display) ---- #
        n = int(s.get("max_spoken_objects"))
        items = []
        for det in ranked[:n]:
            items.append({
                "class_name": det["class_name"],
                "position": det.get("position", "CENTER"),
                "proximity": det.get("proximity", "MEDIUM"),
                "confidence": det.get("confidence", 0.0),
                "urgent": self._is_urgent(det),
            })
        text = generate_multi_message(items, language=language,
                                      bn_overrides=self.bn_overrides)
        self._current_guidance = text
        self._guidance_t = now

        # ---- decide whether to SPEAK ---- #
        speak, speak_reason = self._should_speak(triggering, reason, urgent, text, now)
        if speak:
            key = (triggering["class_name"], triggering.get("position", "CENTER"))
            if key in self._tracked:
                self._tracked[key]["last_announced"] = now
                self._tracked[key]["announced_proximity"] = triggering.get("proximity", "FAR")
                self._tracked[key]["announced_position"] = triggering.get("position", "CENTER")
            self._last_announcement_t = now
            self._last_text = text
            self.announcements += 1
        else:
            self.suppressed += 1

        return {"text": text, "speak": speak, "urgent": urgent,
                "reason": speak_reason, "stale": False, "items": items}

    # ------------------------------------------------------------------ #
    def _should_speak(self, triggering, reason, urgent, text, now) -> tuple[bool, str]:
        """Cooldown + event gating. Returns (speak, human-readable reason)."""
        s = self.settings

        if not s.get("voice_enabled"):
            return False, "voice_disabled"
        if triggering is None:
            return False, "no_new_event"

        # Global chatter limit. Urgent messages get a shortened gap but are
        # never completely exempt, or a NEAR object in CENTER would machine-gun.
        gap = float(s.get("global_min_gap"))
        effective_gap = gap * 0.4 if urgent else gap
        since_any = now - self._last_announcement_t
        if since_any < effective_gap:
            return False, "global_gap:{:.1f}s<{:.1f}s".format(since_any, effective_gap)

        # Per-object cooldown.
        key = (triggering["class_name"], triggering.get("position", "CENTER"))
        last = self._tracked.get(key, {}).get("last_announced", 0.0)
        cooldown = float(s.get("announce_cooldown"))
        since_this = now - last
        if last and since_this < cooldown:
            # A genuine escalation to NEAR overrides the cooldown - that is
            # exactly the moment the user most needs to hear it.
            escalation = reason.startswith("closer") or reason.startswith("moved_into_path")
            if not (urgent and escalation):
                return False, "cooldown:{:.1f}s<{:.1f}s".format(since_this, cooldown)

        # Do not repeat an identical sentence back-to-back.
        if text == self._last_text and since_any < cooldown:
            return False, "duplicate_text"

        return True, reason

    # ------------------------------------------------------------------ #
    def safety_status(self, detections: list[dict]) -> dict:
        """
        A coarse safety level for the interface.

        Derived from the SAME filtered detections and the same danger-class
        list the voice layer uses, so the banner can never contradict what the
        system is saying. Returning a level plus a machine-readable reason lets
        the UI show an icon AND text rather than relying on colour alone,
        which matters for users with low vision or colour blindness.

        Levels, in increasing severity:
            clear     - nothing relevant in view
            attention - something relevant, but not in the path
            warning   - an obstacle is in the path
            critical  - a hazard is in the path and close
        """
        relevant = self._relevant(detections)
        if not relevant:
            return {"level": "clear", "reason": "no_relevant_objects", "object": None}

        ranked = sorted(relevant, key=self.priority_score, reverse=True)
        top = ranked[0]
        cls = top["class_name"]
        centre = top.get("position") == "CENTER"
        near = top.get("proximity") == "NEAR"
        danger = cls in DANGER_CLASSES

        if danger and centre and near:
            level = "critical"
        elif (danger and centre) or (centre and near):
            level = "warning"
        elif centre or near:
            level = "attention"
        else:
            level = "attention"

        return {"level": level,
                "reason": "{}_{}_{}".format(cls, top.get("position", "?").lower(),
                                            top.get("proximity", "?").lower()),
                "object": cls,
                "position": top.get("position"),
                "proximity": top.get("proximity")}

    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        return {
            "tracked_objects": len(self._tracked),
            "announcements": self.announcements,
            "suppressed": self.suppressed,
            "last_announcement_age": (round(time.time() - self._last_announcement_t, 1)
                                      if self._last_announcement_t else None),
        }
