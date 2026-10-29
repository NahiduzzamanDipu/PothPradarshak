#!/usr/bin/env python3
"""
inference.py
============
Run the trained detector on an image, a folder, a video, or a live webcam.
All detection logic lives in detection_core.py; this file is only the CLI and
the display/IO loop, so the navigation module can import detection_core directly
without dragging in any of this.

Usage:
    python scripts/inference.py --source test.jpg
    python scripts/inference.py --source path/to/folder
    python scripts/inference.py --source video.mp4
    python scripts/inference.py --source 0                    # webcam, press q to quit
    python scripts/inference.py --source 0 --save-json out.json
    python scripts/inference.py --source test.jpg --conf 0.4 --nav-preview

Keys during video/webcam: q or ESC = quit, s = save current frame, SPACE = pause.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detection_core import (  # noqa: E402
    Detector, DetectorConfig, draw_detections, summarise_for_navigation,
)

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="YOLO inference: image / folder / video / webcam.")
    ap.add_argument("--source", required=True,
                    help="image path, folder, video path, or a camera index like 0")
    ap.add_argument("--weights", default="runs/train/exp/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-det", type=int, default=50)
    ap.add_argument("--out", default="runs/inference")
    ap.add_argument("--save-json", default=None, help="write all detections to this JSON file")
    ap.add_argument("--no-save", action="store_true", help="do not write annotated output files")
    ap.add_argument("--no-show", action="store_true", help="headless: never open a display window")
    ap.add_argument("--nav-preview", action="store_true",
                    help="print the navigation hand-off summary (LEFT/CENTER/RIGHT + NEAR/MED/FAR)")
    ap.add_argument("--zone-edges", type=float, nargs=2, default=[0.33, 0.66],
                    metavar=("LEFT", "RIGHT"), help="fractional x split points for the three zones")
    return ap.parse_args()


def classify_source(src: str):
    if src.isdigit():
        return "webcam", int(src)
    p = Path(src)
    if not p.exists():
        # allow rtsp:// / http:// streams
        if "://" in src:
            return "video", src
        sys.exit("ERROR: source not found: {}".format(src))
    if p.is_dir():
        return "folder", p
    if p.suffix.lower() in IMAGE_EXTS:
        return "image", p
    if p.suffix.lower() in VIDEO_EXTS:
        return "video", str(p)
    sys.exit("ERROR: unsupported source type: {}".format(p.suffix))


def run_images(det: Detector, paths, args, out_dir: Path) -> list:
    records = []
    for p in paths:
        frame = cv2.imread(str(p))
        if frame is None:
            print("  skip unreadable: {}".format(p))
            continue
        t0 = time.perf_counter()
        dets = det.predict(frame)
        ms = (time.perf_counter() - t0) * 1000

        print("\n{}  ({}x{})  {} detection(s)  {:.0f} ms".format(
            p.name, frame.shape[1], frame.shape[0], len(dets), ms))
        for d in dets:
            print("   {:<18} conf={:.2f}  center=({:.0f},{:.0f})  {}x{}  [{}/{}]".format(
                d["class_name"], d["confidence"], d["x_center"], d["y_center"],
                int(d["width"]), int(d["height"]), d["zone"], d["proximity"]))
        if not dets:
            print("   (nothing above conf={})".format(args.conf))
        if args.nav_preview:
            print("   nav -> {}".format(summarise_for_navigation(dets)))

        records.append({"source": str(p), "width": frame.shape[1],
                        "height": frame.shape[0], "latency_ms": round(ms, 1),
                        "detections": dets})

        vis = draw_detections(frame, dets, cfg=det.cfg)
        if not args.no_save:
            dest = out_dir / "{}_pred.jpg".format(p.stem[:60])
            cv2.imwrite(str(dest), vis)
            print("   saved -> {}".format(dest))
        if not args.no_show and len(paths) == 1:
            cv2.imshow("detection", vis)
            print("\n   press any key to close")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return records


def run_stream(det: Detector, source, args, out_dir: Path, is_webcam: bool) -> list:
    cap = cv2.VideoCapture(source)
    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        sys.exit("ERROR: could not open source: {}\n"
                 "       For a webcam try a different index (0, 1, 2...).".format(source))

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0

    writer = None
    if not args.no_save:
        dest = out_dir / ("webcam_output.mp4" if is_webcam else
                          "{}_pred.mp4".format(Path(str(source)).stem[:50]))
        writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"),
                                 min(src_fps, 30.0), (w, h))
        print("Writing annotated video -> {}".format(dest))

    print("Source {}x{} @ {:.1f} fps{}".format(
        w, h, src_fps, "  ({} frames)".format(total) if total else ""))
    print("Keys: q/ESC quit   s save frame   SPACE pause")

    fps_hist = deque(maxlen=30)
    records, n, paused = [], 0, False

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                n += 1
                t0 = time.perf_counter()
                dets = det.predict(frame)
                dt = time.perf_counter() - t0
                fps_hist.append(1.0 / max(dt, 1e-6))
                fps = sum(fps_hist) / len(fps_hist)

                vis = draw_detections(frame, dets, cfg=det.cfg)
                banner = "FPS {:.1f} | {} obj | {}".format(fps, len(dets), det.device)
                cv2.putText(vis, banner, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(vis, banner, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 1, cv2.LINE_AA)

                if args.save_json:
                    records.append({"frame": n, "latency_ms": round(dt * 1000, 1),
                                    "detections": dets})
                if writer is not None:
                    writer.write(vis)
                if args.nav_preview and dets and n % 10 == 0:
                    print("frame {:>6}  nav -> {}".format(n, summarise_for_navigation(dets)))
                if total and n % 100 == 0:
                    print("  {}/{} frames ({:.0f}%)  {:.1f} FPS".format(n, total, 100.0 * n / total, fps))

            if not args.no_show:
                cv2.imshow("detection - q to quit", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("\nStopped by user.")
                    break
                if key == ord(" "):
                    paused = not paused
                elif key == ord("s"):
                    snap = out_dir / "frame_{:06d}.jpg".format(n)
                    cv2.imwrite(str(snap), vis)
                    print("  saved {}".format(snap))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if fps_hist:
        print("\nProcessed {} frames, average {:.1f} FPS on {}".format(
            n, sum(fps_hist) / len(fps_hist), det.device))
    return records


def main() -> int:
    args = build_args()

    wpath = Path(args.weights)
    if not wpath.is_absolute():
        wpath = ROOT / wpath
    if not wpath.exists():
        print("ERROR: weights not found: {}".format(wpath))
        print("       Train a model first: python scripts/train.py")
        print("       (or pass --weights yolo11n.pt to try the COCO-pretrained model,")
        print("        which knows person/car/chair/etc. but nothing Bangladesh-specific)")
        return 2

    cfg = DetectorConfig(
        weights=str(wpath), conf=args.conf, iou=args.iou, imgsz=args.imgsz,
        device=args.device, max_det=args.max_det,
        zone_left_edge=args.zone_edges[0], zone_right_edge=args.zone_edges[1],
    )
    det = Detector(cfg)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    kind, src = classify_source(args.source)
    print("source type: {}".format(kind))

    if kind == "image":
        records = run_images(det, [src], args, out_dir)
    elif kind == "folder":
        paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not paths:
            print("ERROR: no images in {}".format(src))
            return 2
        print("{} images".format(len(paths)))
        records = run_images(det, paths, args, out_dir)
    else:
        records = run_stream(det, src, args, out_dir, is_webcam=(kind == "webcam"))

    if args.save_json:
        dest = Path(args.save_json)
        if not dest.is_absolute():
            dest = out_dir / dest
        dest.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print("Detections written to {}".format(dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
