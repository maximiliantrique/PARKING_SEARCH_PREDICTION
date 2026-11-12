"""Rend le paquet `src/parking_search_prediction` importable depuis `scripts/`."""

from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> Path:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src
