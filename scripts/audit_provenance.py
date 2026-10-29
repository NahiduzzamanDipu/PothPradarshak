#!/usr/bin/env python3
"""
audit_provenance.py
===================
The `image/` folder is a merge of several unrelated source datasets. This script
groups every file by filename family + resolution so the composition of the merge
can be reported factually, and extracts the class-name tokens embedded in the
Roboflow-exported filenames (they hint at the original datasets' label sets).

Usage:  python scripts/audit_provenance.py
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "image"
OUT = ROOT / "reports"

RE_NUM_PNG = re.compile(r"^\d+\.png$", re.I)
RE_NUM_JPG = re.compile(r"^\d+\.jpg$", re.I)
RE_RANDOM8 = re.compile(r"^[a-z0-9]{8}\.jpe?g$")
RE_RF = re.compile(r"\.rf\.[0-9a-f]+", re.I)


def size_of(p: str):
    try:
        with Image.open(p) as im:
            return p, im.size[0], im.size[1]
    except Exception:
        return p, -1, -1


def group_of(name: str, w: int, h: int) -> str:
    mx = max(w, h)
    if RE_NUM_PNG.match(name) and mx <= 250:
        return "1. GTSRB German traffic-sign CROPS (.png, tiny)"
    if RE_NUM_JPG.match(name) and (w, h) == (256, 256):
        return "2. Dhaka footpath scenes (BD, 256x256)"
    if name.startswith("gambar-"):
        return "3. Indonesian sidewalk set 'gambar-*' (608x608)"
    if name.startswith(("Dipto_", "Dipto-", "Asraf_")):
        return "4. Self-collected 'Dipto_/Asraf_' (BD)"
    if RE_RF.search(name):
        return "5. Other Roboflow exports (class-named)"
    if RE_RANDOM8.match(name):
        return "6. Scraped web/stock images (random 8-char names)"
    if RE_NUM_JPG.match(name) or RE_NUM_PNG.match(name):
        return "7. Numbered .jpg, non-256 (mixed)"
    return "8. Uncategorised"


def rf_class_token(name: str) -> str | None:
    """Roboflow keeps the original filename before '.rf.'; its leading token is
    usually the source dataset's class or folder name."""
    if not RE_RF.search(name):
        return None
    stem = RE_RF.split(name)[0]
    stem = re.sub(r"_(jpg|jpeg|png|JPG)$", "", stem)
    tok = re.split(r"[-_ ]", stem)[0]
    tok = re.sub(r"\d+$", "", tok)
    return tok.lower() if tok and not tok.isdigit() else None


def main() -> None:
    files = [p for p in IMG.iterdir() if p.is_file()]
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(size_of, [str(p) for p in files], chunksize=128))

    groups = Counter()
    group_res = defaultdict(Counter)
    group_px = defaultdict(list)
    rf_tokens = Counter()

    for path, w, h in results:
        name = Path(path).name
        g = group_of(name, w, h)
        groups[g] += 1
        group_res[g][(w, h)] += 1
        group_px[g].append(max(w, h))
        t = rf_class_token(name)
        if t:
            rf_tokens[t] += 1

    L = []
    A = L.append
    A("=" * 78)
    A("DATASET PROVENANCE AUDIT  -  what is actually inside image/")
    A("=" * 78)
    A("Total files: {:,}".format(len(files)))
    A("")
    A("{:<50} {:>7} {:>7}  {}".format("GROUP", "COUNT", "SHARE", "TYPICAL SIZE"))
    A("-" * 78)
    for g, c in sorted(groups.items()):
        top = group_res[g].most_common(1)[0]
        A("{:<50} {:>7,} {:>6.1f}%  {}x{} ({:,} imgs)".format(
            g, c, 100.0 * c / len(files), top[0][0], top[0][1], top[1]))
    A("-" * 78)
    A("")
    A("Per-group resolution detail (top 4 sizes each):")
    for g in sorted(groups):
        A("  " + g)
        for (w, h), c in group_res[g].most_common(4):
            A("      {:>5}x{:<5} {:>7,}".format(w, h, c))
    A("")
    A("=" * 78)
    A("CLASS-NAME TOKENS EMBEDDED IN ROBOFLOW FILENAMES")
    A("(these are the ORIGINAL datasets' class/folder names - NOT labels,")
    A(" but the strongest available evidence of what was intended)")
    A("=" * 78)
    for tok, c in rf_tokens.most_common(40):
        A("    {:<28} {:>6,}".format(tok, c))
    A("")
    A("Distinct tokens: {}".format(len(rf_tokens)))

    text = "\n".join(L)
    print(text)
    OUT.mkdir(exist_ok=True)
    (OUT / "provenance_audit.txt").write_text(text, encoding="utf-8")
    print("\nSaved: {}".format(OUT / "provenance_audit.txt"))


if __name__ == "__main__":
    main()
