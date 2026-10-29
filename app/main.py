#!/usr/bin/env python3
"""
app/main.py
===========
Entry point for the real-time application:

    python app/main.py

This is a thin launcher. All application code lives in backend/, which is put
on sys.path here so its modules can import each other by plain name
(`from config import ...`) whether they are started from this file, from
`python backend/main.py`, or from uvicorn directly.

Useful flags (full list: python app/main.py --help):

    --camera 1              use a different camera index, or a phone stream URL
    --model models/best.pt  force specific weights instead of auto-discovery
    --language en           start in English (default is Bengali)
    --device cpu            force CPU even when CUDA is present
    --host 0.0.0.0          reachable from a phone on the same Wi-Fi
    --autostart             begin detection without pressing Start Camera
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if not BACKEND_DIR.is_dir():
    raise SystemExit(
        "Cannot find the backend/ directory next to app/.\n"
        "Expected: {}".format(BACKEND_DIR)
    )

# Prepend so the project's own modules always win over a same-named package
# that happens to be installed in site-packages.
sys.path.insert(0, str(BACKEND_DIR))

from main import main  # noqa: E402  (import must follow the sys.path setup)

if __name__ == "__main__":
    raise SystemExit(main())
