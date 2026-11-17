from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "weights" / "best.pt"
MODEL_PATH = Path(os.getenv("ROAD_DAMAGE_MODEL", str(DEFAULT_MODEL)))
CLASS_NAMES = {
    0: "D00 Longitudinal crack",
    1: "D10 Transverse crack",
    2: "D20 Alligator crack",
    3: "D40 Pothole",
}
