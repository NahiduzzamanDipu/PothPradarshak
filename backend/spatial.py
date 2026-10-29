#!/usr/bin/env python3
"""
spatial.py
==========
Turns a bounding box into the two spatial facts the navigation layer needs:

    position   LEFT | CENTER | RIGHT     (from the box centre's x fraction)
    proximity  NEAR | MEDIUM | FAR       (from the box's area fraction)

=============================================================================
HONEST LIMITATION - state this in the thesis and the viva
=============================================================================
`proximity` is APPROXIMATE PROXIMITY ESTIMATION, not distance measurement.

It is derived from apparent box size in a single RGB image. A bus 30 m away can
occupy more pixels than a chair 2 m away, so the estimate is only comparable
BETWEEN objects of similar real-world size, never across classes.

A monocular RGB camera cannot recover metric distance without extra
information. Adding any one of these would make it a real measurement:
  * a monocular depth network (MiDaS, Depth-Anything, ZoeDepth)
  * a stereo pair or a phone ToF/LiDAR sensor
  * known camera height + a ground-plane assumption, for objects touching
    the floor (workable here, since footpath obstacles do touch the ground)

The class is written so a depth source can be dropped in: implement
`DepthProvider.depth_at()` and pass it to `SpatialAnalyzer`, and proximity will
be computed from real depth instead of box area, with no other code changing.
=============================================================================
"""

from __future__ import annotations

from typing import Protocol

LEFT, CENTER, RIGHT = "LEFT", "CENTER", "RIGHT"
NEAR, MEDIUM, FAR = "NEAR", "MEDIUM", "FAR"

# Ordering used to decide whether proximity got worse (FAR -> NEAR is an event).
PROXIMITY_RANK = {FAR: 0, MEDIUM: 1, NEAR: 2}


class DepthProvider(Protocol):
    """Extension point. Implement this to upgrade to genuine distance."""

    def depth_at(self, cx: float, cy: float, frame_w: int, frame_h: int) -> float | None:
        """Return metres at that pixel, or None when unavailable."""
        ...


class SpatialAnalyzer:
    """
    All thresholds are configurable (sections 9, 10, 21) and are re-read from
    the Settings object on every call, so the UI sliders take effect live.
    """

    def __init__(self, settings, depth_provider: DepthProvider | None = None):
        self.settings = settings
        self.depth_provider = depth_provider

        # Metre thresholds used only when a real depth provider exists.
        self.depth_near_m = 1.5
        self.depth_medium_m = 4.0

    # ------------------------------------------------------------------ #
    def get_position(self, center_x: float, frame_width: int) -> str:
        """
        LEFT / CENTER / RIGHT from the horizontal centre of the box.

            0%            left_edge      right_edge          100%
            |---- LEFT ------|--- CENTER ---|---- RIGHT ------|
        """
        if frame_width <= 0:
            return CENTER
        frac = center_x / float(frame_width)
        s = self.settings
        if frac < s.get("zone_left_edge"):
            return LEFT
        if frac > s.get("zone_right_edge"):
            return RIGHT
        return CENTER

    # ------------------------------------------------------------------ #
    def get_proximity(self, bbox, frame_width: int, frame_height: int) -> tuple[str, float]:
        """
        APPROXIMATE proximity from box area relative to frame area.
        Returns (bucket, area_ratio).

        If a DepthProvider is attached, real depth wins and area is ignored.
        """
        x1, y1, x2, y2 = bbox
        bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
        frame_area = float(max(frame_width * frame_height, 1))
        area_ratio = (bw * bh) / frame_area

        if self.depth_provider is not None:
            cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
            metres = self.depth_provider.depth_at(cx, cy, frame_width, frame_height)
            if metres is not None:
                if metres <= self.depth_near_m:
                    return NEAR, area_ratio
                if metres <= self.depth_medium_m:
                    return MEDIUM, area_ratio
                return FAR, area_ratio

        s = self.settings
        if area_ratio >= s.get("near_area_ratio"):
            return NEAR, area_ratio
        if area_ratio >= s.get("medium_area_ratio"):
            return MEDIUM, area_ratio
        return FAR, area_ratio

    # ------------------------------------------------------------------ #
    def annotate(self, detection: dict, frame_width: int, frame_height: int) -> dict:
        """Add position / proximity / area_ratio to a detection dict in place."""
        cx = detection["center"][0]
        detection["position"] = self.get_position(cx, frame_width)
        proximity, area_ratio = self.get_proximity(
            detection["bbox"], frame_width, frame_height)
        detection["proximity"] = proximity
        detection["area_ratio"] = round(area_ratio, 5)
        return detection

    # ------------------------------------------------------------------ #
    def zone_boundaries_px(self, frame_width: int) -> tuple[int, int]:
        """Pixel x of the two dividing lines, for drawing the overlay."""
        s = self.settings
        return (int(s.get("zone_left_edge") * frame_width),
                int(s.get("zone_right_edge") * frame_width))
