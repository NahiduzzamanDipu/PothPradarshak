#!/usr/bin/env python3
"""
review_labels.py
================
STEP 2 of the annotation workflow: a human verifies every candidate label.

    build_annotation_set.py -> autolabel.py -> review_labels.py -> prepare_dataset.py

Starts a small local web app for reviewing and correcting the machine-generated
boxes from autolabel.py. Nothing here is automatic: an image counts as
verified only when YOU press a key on it.

    python scripts/review_labels.py
    -> open http://127.0.0.1:8009

WHY A CUSTOM TOOL RATHER THAN ROBOFLOW / LABELIMG
-------------------------------------------------
Both are fine, and `--export-only` will hand you a folder ready to upload
anywhere. But the bottleneck for this thesis is REVIEW SPEED across ~2,800
images, and general tools optimise for drawing from scratch. This one is built
around the actual task: most images already have candidate boxes, so the common
action is "yes, correct" (one key) or "delete that box" (one click), not
drawing. Hardest classes are served first, so if review time runs out, the
classes that matter most for navigation are the ones that got done.

KEYBOARD (the whole point - keep both hands on the keyboard)
------------------------------------------------------------
    A / Enter    accept image as correct, go to next
    D            delete the currently selected box
    1-9          set the class of the selected box
    N            skip without verifying (decide later)
    Backspace    previous image
    S            save now (also saves automatically on every action)
    Drag on the image to draw a new box; click a box to select it.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_IMAGES = ROOT / "annotation_input"
DEFAULT_LABELS = ROOT / "autolabels"
EXPORT_DIR = ROOT / "annotated"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
def load_classes() -> tuple[list[str], list[str]]:
    """Class names from data.yaml - the single source of truth for ids."""
    import yaml
    cfg = yaml.safe_load((ROOT / "data.yaml").read_text(encoding="utf-8"))
    names = cfg.get("names") or {}
    names_bn = cfg.get("names_bn") or {}
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    if isinstance(names_bn, list):
        names_bn = {i: n for i, n in enumerate(names_bn)}
    n = max(int(k) for k in names) + 1
    return ([str(names.get(i, i)) for i in range(n)],
            [str(names_bn.get(i, "")) for i in range(n)])


class ReviewStore:
    """Owns the label files and the verified/skipped state on disk."""

    def __init__(self, images_dir: Path, labels_root: Path):
        self.images_dir = images_dir
        self.labels_dir = labels_root / "labels"
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = labels_root / "review_state.json"
        self.manifest_path = labels_root / "manifest.csv"

        self.images = sorted(p.name for p in images_dir.iterdir()
                             if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if not self.images:
            raise SystemExit("No images in {}".format(images_dir))

        self.state = self._load_state()
        self.order = self._build_order()

    # ------------------------------------------------------------------ #
    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"verified": {}, "skipped": {}}

    def save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ #
    def _build_order(self) -> list[str]:
        """
        Review order = hardest and most important first.

        Rationale: review time is the scarce resource. The classes the
        navigation layer treats as dangerous (pothole, open_drain, curb) are
        exactly the ones the auto-labeller is worst at, so they must not be
        left to a session that never happens. Images with no candidate boxes
        come next (likely misses), then everything else.
        """
        priority: dict[str, int] = {}
        if self.manifest_path.exists():
            with self.manifest_path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    name = row["image"]
                    classes = row.get("classes", "") or ""
                    n_boxes = int(row.get("n_boxes", 0) or 0)
                    if any(c in classes for c in ("pothole", "open_drain", "curb")):
                        priority[name] = 0
                    elif n_boxes == 0:
                        priority[name] = 1
                    elif row.get("needs_attention") == "1":
                        priority[name] = 2
                    else:
                        priority[name] = 3
        return sorted(self.images, key=lambda n: (priority.get(n, 3), n))

    # ------------------------------------------------------------------ #
    def label_path(self, image_name: str) -> Path:
        return self.labels_dir / (Path(image_name).stem + ".txt")

    def read_boxes(self, image_name: str) -> list[dict]:
        p = self.label_path(image_name)
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                out.append({
                    "cls": int(float(parts[0])),
                    "cx": float(parts[1]), "cy": float(parts[2]),
                    "w": float(parts[3]), "h": float(parts[4]),
                })
            except ValueError:
                continue
        return out

    def write_boxes(self, image_name: str, boxes: list[dict]) -> None:
        lines = []
        for b in boxes:
            cx = min(max(float(b["cx"]), 0.0), 1.0)
            cy = min(max(float(b["cy"]), 0.0), 1.0)
            w = min(max(float(b["w"]), 0.0), 1.0)
            h = min(max(float(b["h"]), 0.0), 1.0)
            if w <= 0.001 or h <= 0.001:
                continue
            lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(int(b["cls"]), cx, cy, w, h))
        self.label_path(image_name).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        verified = len(self.state["verified"])
        return {
            "total": len(self.images),
            "verified": verified,
            "skipped": len(self.state["skipped"]),
            "remaining": len(self.images) - verified,
            "percent": round(100.0 * verified / max(len(self.images), 1), 1),
        }


# --------------------------------------------------------------------------- #
def export(store: ReviewStore, dest: Path, verified_only: bool = True) -> int:
    """Copy verified image+label pairs into annotated/ for prepare_dataset.py."""
    img_out = dest / "images"
    lbl_out = dest / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n = 0
    for name in store.images:
        if verified_only and name not in store.state["verified"]:
            continue
        src_img = store.images_dir / name
        src_lbl = store.label_path(name)
        if not src_img.exists():
            continue
        shutil.copy2(src_img, img_out / name)
        # A verified image with no objects still needs an empty label file:
        # YOLO treats it as a valid background example.
        if src_lbl.exists():
            shutil.copy2(src_lbl, lbl_out / src_lbl.name)
        else:
            (lbl_out / (Path(name).stem + ".txt")).write_text("", encoding="utf-8")
        n += 1

    print("Exported {} {} image/label pairs to {}".format(
        n, "verified" if verified_only else "total", dest))
    if n == 0:
        print("\nNothing exported. Verify some images first:")
        print("  python scripts/review_labels.py")
    else:
        print("\nNEXT STEP\n  python scripts/prepare_dataset.py")
    return n


# --------------------------------------------------------------------------- #
# Request models MUST live at module scope.
#
# `from __future__ import annotations` turns every annotation into a string,
# and FastAPI resolves a handler's type hints against its MODULE globals. A
# BaseModel defined inside build_app() is invisible there, so FastAPI cannot
# tell that `req: SaveReq` is a body model and silently demotes it to a query
# parameter - every save then fails with "field required". Keep these here.
# --------------------------------------------------------------------------- #
class SaveReq(BaseModel):
    image: str
    boxes: list[dict]
    verified: bool = True


# --------------------------------------------------------------------------- #
def build_app(store: ReviewStore, class_names: list[str], class_names_bn: list[str]):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    app = FastAPI(title="Label Review")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (Path(__file__).parent / "review_ui.html").read_text(encoding="utf-8")

    @app.get("/api/classes")
    def classes():
        return {"names": class_names, "names_bn": class_names_bn}

    @app.get("/api/list")
    def list_images():
        return {
            "images": store.order,
            "verified": list(store.state["verified"].keys()),
            "skipped": list(store.state["skipped"].keys()),
            "stats": store.stats(),
        }

    @app.get("/api/image/{name}")
    def image(name: str):
        p = store.images_dir / name
        if not p.exists() or p.parent != store.images_dir:
            raise HTTPException(404, "not found")
        return FileResponse(str(p))

    @app.get("/api/boxes/{name}")
    def boxes(name: str):
        return {
            "image": name,
            "boxes": store.read_boxes(name),
            "verified": name in store.state["verified"],
        }

    @app.post("/api/save")
    def save(req: SaveReq):
        if req.image not in set(store.images):
            raise HTTPException(404, "unknown image")
        store.write_boxes(req.image, req.boxes)
        if req.verified:
            store.state["verified"][req.image] = True
            store.state["skipped"].pop(req.image, None)
        store.save_state()
        return {"ok": True, "stats": store.stats()}

    @app.post("/api/skip/{name}")
    def skip(name: str):
        store.state["skipped"][name] = True
        store.save_state()
        return {"ok": True, "stats": store.stats()}

    @app.post("/api/export")
    def do_export():
        n = export(store, EXPORT_DIR, verified_only=True)
        return {"ok": True, "exported": n, "dest": str(EXPORT_DIR)}

    return app


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Review and correct auto-generated labels.")
    ap.add_argument("--images", default=None, help="image folder (default: annotation_input/)")
    ap.add_argument("--labels", default=None, help="autolabel output (default: autolabels/)")
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--export-only", action="store_true",
                    help="export verified pairs to annotated/ and exit")
    ap.add_argument("--export-all", action="store_true",
                    help="export EVERY image including unverified (not recommended)")
    ap.add_argument("--stats", action="store_true", help="print progress and exit")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    images_dir = Path(args.images) if args.images else DEFAULT_IMAGES
    labels_root = Path(args.labels) if args.labels else DEFAULT_LABELS

    if not images_dir.is_dir():
        raise SystemExit("Image folder not found: {}\nRun scripts/build_annotation_set.py first."
                         .format(images_dir))

    store = ReviewStore(images_dir, labels_root)
    class_names, class_names_bn = load_classes()

    if args.stats:
        s = store.stats()
        print("=" * 60)
        print("REVIEW PROGRESS")
        print("=" * 60)
        print("  total images : {}".format(s["total"]))
        print("  verified     : {}  ({}%)".format(s["verified"], s["percent"]))
        print("  skipped      : {}".format(s["skipped"]))
        print("  remaining    : {}".format(s["remaining"]))
        print("=" * 60)
        return 0

    if args.export_only or args.export_all:
        export(store, EXPORT_DIR, verified_only=not args.export_all)
        return 0

    s = store.stats()
    print("=" * 70)
    print("LABEL REVIEW TOOL")
    print("=" * 70)
    print("  images    : {}".format(s["total"]))
    print("  verified  : {} ({}%)".format(s["verified"], s["percent"]))
    print("  remaining : {}".format(s["remaining"]))
    print()
    print("  Ordered hardest-first: pothole / open_drain / curb candidates and")
    print("  images with no boxes come before the easy ones.")
    print()
    print("  Open http://{}:{}".format(args.host, args.port))
    print("  Keys: A accept | D delete box | 1-9 class | N skip | Backspace back")
    print("=" * 70)

    import uvicorn
    app = build_app(store, class_names, class_names_bn)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
