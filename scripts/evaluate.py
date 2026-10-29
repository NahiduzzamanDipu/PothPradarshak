#!/usr/bin/env python3
"""
evaluate.py
===========
Evaluate a trained model on the TEST split (held out, never seen during training
or early stopping) and write thesis-ready metrics and plots.

Ultralytics generates the standard plots automatically into the run folder:
    BoxPR_curve.png, BoxP_curve.png, BoxR_curve.png, BoxF1_curve.png,
    confusion_matrix.png, confusion_matrix_normalized.png
Training loss curves live in the TRAINING run folder as results.png / results.csv.

This script additionally writes:
    metrics.json            machine-readable results
    metrics_report.txt      per-class table + metric explanations (EN + BN)
    per_class_map.png       per-class mAP@50 / mAP@50:95 bar chart
    speed_benchmark.txt     measured latency and FPS on this machine

Usage:
    python scripts/evaluate.py --weights runs/train/exp/weights/best.pt
    python scripts/evaluate.py --weights ... --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import resolve_for_ultralytics  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

EXPLANATION = """
================================================================================
WHAT THESE METRICS MEAN  /  এই মেট্রিকগুলোর অর্থ
================================================================================

IoU (Intersection over Union)
  EN: Overlap between the predicted box and the ground-truth box, computed as
      (area of intersection) / (area of union). 1.0 = perfect overlap, 0 = none.
      A prediction only counts as correct if its IoU passes a threshold.
  BN: IoU হলো মডেলের আঁকা বক্স এবং প্রকৃত (ground-truth) বক্সের মধ্যে কতটুকু মিল
      আছে তার পরিমাপ। দুটি বক্সের সাধারণ অংশকে (intersection) মোট অংশ দিয়ে
      (union) ভাগ করা হয়। ১.০ মানে পুরোপুরি মিল, ০ মানে কোনো মিল নেই।

Precision (P)  =  TP / (TP + FP)
  EN: Of everything the model claimed to see, what fraction was really there?
      Low precision = false alarms.
  BN: মডেল যতগুলো বস্তু শনাক্ত করেছে, তার মধ্যে কতগুলো আসলেই সঠিক ছিল।
      Precision কম হলে মডেল ভুল জিনিস দেখে — অর্থাৎ মিথ্যা সতর্কতা দেয়।
  Why it matters here: a false "obstacle ahead" makes a blind user stop for
  nothing. Repeated false alarms destroy trust in the device.
  এখানে গুরুত্ব: ভুল সতর্কবার্তা ব্যবহারকারীকে অকারণে থামিয়ে দেয় এবং বারবার
  ভুল হলে ব্যবহারকারী যন্ত্রটির উপর আস্থা হারিয়ে ফেলেন।

Recall (R)  =  TP / (TP + FN)
  EN: Of everything that was actually there, what fraction did the model find?
      Low recall = missed objects.
  BN: বাস্তবে যতগুলো বস্তু ছিল, মডেল তার মধ্যে কতগুলো খুঁজে পেয়েছে।
      Recall কম হলে মডেল অনেক বস্তু দেখতেই পায় না।
  Why it matters here: a MISSED open manhole or pole is a physical injury.
  For this application recall is the more safety-critical of the two.
  এখানে গুরুত্ব: একটি খোলা ম্যানহোল বা খুঁটি শনাক্ত করতে না পারলে ব্যবহারকারী
  আঘাত পেতে পারেন। তাই এই কাজে recall বেশি নিরাপত্তা-গুরুত্বপূর্ণ।

mAP@50  (mean Average Precision at IoU = 0.50)
  EN: Average Precision is the area under the precision-recall curve for one
      class; mAP is the mean of that over all classes, counting a detection as
      correct when IoU >= 0.50. A lenient, forgiving measure of "did you find
      the object roughly in the right place?"
  BN: প্রতিটি ক্লাসের precision-recall কার্ভের নিচের ক্ষেত্রফল হলো Average
      Precision; সব ক্লাসের গড় হলো mAP। IoU ০.৫০ বা তার বেশি হলে শনাক্তকরণ
      সঠিক ধরা হয়। এটি তুলনামূলক সহজ মানদণ্ড — বস্তুটি মোটামুটি সঠিক জায়গায়
      পাওয়া গেছে কিনা তা দেখে।

mAP@50:95
  EN: The same quantity averaged over ten IoU thresholds (0.50, 0.55, ... 0.95).
      Much stricter: it rewards tightly, accurately placed boxes. This is the
      primary COCO metric and the one to headline in the thesis.
  BN: একই পরিমাপ, তবে দশটি ভিন্ন IoU সীমার (০.৫০ থেকে ০.৯৫) গড়। এটি অনেক
      কঠোর — বক্সটি কতটা নিখুঁতভাবে বসেছে তাও বিবেচনা করে। এটিই COCO-এর প্রধান
      মেট্রিক এবং থিসিসে এটিকেই মূল ফলাফল হিসেবে দেখানো উচিত।
  Rule of thumb: mAP@50:95 is always lower than mAP@50. A large gap between
  them means the classes are found but the boxes are loosely positioned.
  মনে রাখুন: mAP@50:95 সবসময় mAP@50 এর চেয়ে কম হয়। দুটির পার্থক্য বেশি হলে
  বুঝতে হবে ক্লাস ঠিক আছে কিন্তু বক্সের অবস্থান নিখুঁত নয়।

Confusion matrix
  EN: A grid whose rows are predictions and columns are true classes. The
      diagonal is correct; off-diagonal cells show which classes get mixed up.
      The extra "background" row/column captures false alarms (background
      predicted as an object) and misses (object predicted as background).
  BN: একটি ছক যেখানে সারিতে মডেলের ভবিষ্যদ্বাণী এবং কলামে প্রকৃত ক্লাস থাকে।
      কর্ণ (diagonal) বরাবর সংখ্যাগুলো সঠিক; কর্ণের বাইরের ঘরগুলো দেখায় কোন
      ক্লাসের সাথে কোন ক্লাস গুলিয়ে ফেলা হচ্ছে। অতিরিক্ত "background" সারি ও
      কলাম মিথ্যা শনাক্তকরণ ও বাদ পড়া বস্তু দেখায়।

Precision-Recall curve
  EN: Sweeping the confidence threshold from high to low traces precision
      against recall. It shows the trade-off you must choose at deployment:
      raise the threshold for fewer false alarms, lower it to miss less.
  BN: কনফিডেন্স থ্রেশহোল্ড বাড়ালে-কমালে precision ও recall কীভাবে বদলায় তা এই
      কার্ভ দেখায়। ব্যবহারের সময় এই ভারসাম্যটি বেছে নিতে হয়: থ্রেশহোল্ড বাড়ালে
      মিথ্যা সতর্কতা কমে, কমালে বস্তু বাদ পড়া কমে।

Loss (box / cls / dfl)
  EN: box_loss - how wrong the box coordinates are. cls_loss - how wrong the
      class label is. dfl_loss - distribution focal loss, refining box edges.
      Training loss falling while VALIDATION loss rises means overfitting.
  BN: box_loss — বক্সের অবস্থান কতটা ভুল। cls_loss — ক্লাস শনাক্তকরণ কতটা ভুল।
      dfl_loss — বক্সের প্রান্ত আরও নিখুঁত করার জন্য ব্যবহৃত loss। ট্রেনিং loss
      কমতে থাকলেও validation loss বাড়তে থাকলে বুঝতে হবে মডেল overfitting করছে।
================================================================================
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a trained YOLO model.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.001, help="low conf is correct for mAP computation")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU during evaluation")
    ap.add_argument("--device", default=None)
    ap.add_argument("--project", default="runs/evaluate")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--benchmark", action="store_true", help="also measure inference latency / FPS")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    wpath = Path(args.weights)
    if not wpath.exists():
        print("ERROR: weights not found: {}".format(wpath))
        print("Train a model first: python scripts/train.py")
        return 2

    data_path = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not cfg.get(args.split):
        print("ERROR: data.yaml declares no '{}' split.".format(args.split))
        return 2

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating {} on the '{}' split (device={}) ...".format(wpath.name, args.split, device))

    model = YOLO(str(wpath))
    resolved_data = resolve_for_ultralytics(data_path)
    m = model.val(
        data=str(resolved_data), split=args.split, imgsz=args.imgsz, batch=args.batch,
        conf=args.conf, iou=args.iou, device=device,
        project=args.project, name=args.name, exist_ok=True, plots=True, verbose=True,
    )

    save_dir = Path(getattr(m, "save_dir", Path(args.project) / args.name))
    names = model.names
    box = m.box

    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "fitness": float(getattr(m, "fitness", float("nan"))),
    }

    per_class = []
    ap50 = np.atleast_1d(np.asarray(box.ap50, dtype=float))
    ap = np.atleast_1d(np.asarray(box.ap, dtype=float))
    p = np.atleast_1d(np.asarray(box.p, dtype=float))
    r = np.atleast_1d(np.asarray(box.r, dtype=float))
    for i, cid in enumerate(box.ap_class_index):
        per_class.append({
            "class_id": int(cid),
            "class_name": names.get(int(cid), str(int(cid))),
            "precision": float(p[i]) if i < len(p) else float("nan"),
            "recall": float(r[i]) if i < len(r) else float("nan"),
            "mAP50": float(ap50[i]) if i < len(ap50) else float("nan"),
            "mAP50_95": float(ap[i]) if i < len(ap) else float("nan"),
        })

    # -------------------------- text report --------------------------- #
    L = []
    A = L.append
    A("=" * 80)
    A("EVALUATION REPORT")
    A("=" * 80)
    A("weights : {}".format(wpath))
    A("data    : {}".format(data_path))
    A("split   : {}   imgsz: {}   device: {}".format(args.split, args.imgsz, device))
    A("")
    A("OVERALL")
    A("-" * 80)
    A("  Precision   : {:.4f}".format(overall["precision"]))
    A("  Recall      : {:.4f}".format(overall["recall"]))
    A("  mAP@50      : {:.4f}".format(overall["mAP50"]))
    A("  mAP@50:95   : {:.4f}".format(overall["mAP50_95"]))
    A("")
    A("PER-CLASS")
    A("-" * 80)
    A("{:>4} {:<24} {:>10} {:>10} {:>10} {:>12}".format(
        "ID", "CLASS", "P", "R", "mAP@50", "mAP@50:95"))
    for c in sorted(per_class, key=lambda d: -d["mAP50_95"]):
        A("{:>4} {:<24} {:>10.4f} {:>10.4f} {:>10.4f} {:>12.4f}".format(
            c["class_id"], c["class_name"], c["precision"], c["recall"], c["mAP50"], c["mAP50_95"]))
    A("")
    weak = [c["class_name"] for c in per_class if c["mAP50"] < 0.30]
    if weak:
        A("Classes below 0.30 mAP@50: {}".format(weak))
        A("  Usually caused by too few training examples, inconsistent labelling,")
        A("  or objects that are too small at this imgsz. Investigate before")
        A("  reporting - do not just present the overall average.")
        A("")
    A("PLOTS GENERATED BY ULTRALYTICS IN {}".format(save_dir))
    A("  BoxPR_curve.png ... precision vs recall")
    A("  BoxP_curve.png / BoxR_curve.png / BoxF1_curve.png ... vs confidence threshold")
    A("  confusion_matrix.png, confusion_matrix_normalized.png")
    A("  (training/validation LOSS curves are in the TRAINING folder: results.png)")
    A(EXPLANATION)

    text = "\n".join(L)
    print("\n" + text)
    (save_dir / "metrics_report.txt").write_text(text, encoding="utf-8")
    (save_dir / "metrics.json").write_text(json.dumps(
        {"weights": str(wpath), "split": args.split, "imgsz": args.imgsz,
         "overall": overall, "per_class": per_class}, indent=2), encoding="utf-8")

    # -------------------------- per-class chart ----------------------- #
    if per_class:
        s = sorted(per_class, key=lambda d: -d["mAP50_95"])
        labels = [c["class_name"] for c in s]
        x = np.arange(len(s))
        fig, ax = plt.subplots(figsize=(max(7, len(s) * 0.7), 5))
        ax.bar(x - 0.2, [c["mAP50"] for c in s], 0.4, label="mAP@50", color="#3b82f6")
        ax.bar(x + 0.2, [c["mAP50_95"] for c in s], 0.4, label="mAP@50:95", color="#f59e0b")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(0, 1); ax.set_ylabel("score")
        ax.set_title("Per-class performance ({} split)".format(args.split))
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(save_dir / "per_class_map.png", dpi=150); plt.close(fig)

    # -------------------------- speed benchmark ----------------------- #
    if args.benchmark:
        import time
        dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
        for _ in range(10):
            model.predict(dummy, imgsz=args.imgsz, device=device, verbose=False)
        t0 = time.perf_counter()
        N = 50
        for _ in range(N):
            model.predict(dummy, imgsz=args.imgsz, device=device, verbose=False)
        dt = (time.perf_counter() - t0) / N
        bench = ("SPEED BENCHMARK\n"
                 "  device       : {}\n"
                 "  imgsz        : {}\n"
                 "  latency      : {:.1f} ms/image\n"
                 "  throughput   : {:.1f} FPS\n"
                 "  (single image, batch=1, includes pre/post-processing)\n"
                 "  Real-time assistive use generally needs >= 10 FPS.\n"
                 ).format(device, args.imgsz, dt * 1000, 1.0 / dt)
        print("\n" + bench)
        (save_dir / "speed_benchmark.txt").write_text(bench, encoding="utf-8")

    print("Saved: {}".format(save_dir / "metrics_report.txt"))
    print("Saved: {}".format(save_dir / "metrics.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
