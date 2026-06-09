# Author: X.M. Christine Zhu

"""
Experiment 1 — Data collection: Tracker characterization.

Goal
----
Measure how Atracsys noise varies across the workspace.

Procedure
---------
1. Place the Anatomy rigid body at POSITION 1 and hold still.
2. Collect 200 measurements  →  saved to data/pos_01.csv
3. Move BodyA to POSITION 2, repeat.
4. Continue for all positions (vary distance, angle from camera).

Result files
------------
data/pos_XX.csv   — (200, 16) CSV, each row is a flattened 4×4 pose matrix
data/positions.txt — one line per position: label, distance_mm, angle_deg
"""

import sys
import pathlib
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_EXP  = _HERE.parent
sys.path.insert(0, str(_EXP))

from utils.atracsys_interface import AtracsysTracker
from utils.se3_stats import save_poses_csv

# ── Configuration ─────────────────────────────────────────────────────────────

BODY_NAME   = "Anatomy"    # 4-marker rigid body — easy to move by hand
N_SAMPLES   = 200          # measurements per position
DATA_DIR    = _HERE / "data"

# Label each position you will physically move the body to.
# Fill in approx distance (mm) and angle (deg from camera optical axis) after
# you set up each position — used only for the analysis labels.
POSITIONS = [
    {"label": "pos_01", "distance_mm": 500,  "angle_deg":  0},
    {"label": "pos_02", "distance_mm": 700,  "angle_deg":  0},
    {"label": "pos_03", "distance_mm": 900,  "angle_deg":  0},
    {"label": "pos_04", "distance_mm": 1100, "angle_deg":  0},
    {"label": "pos_05", "distance_mm": 700,  "angle_deg": 15},
    {"label": "pos_06", "distance_mm": 700,  "angle_deg": 30},
    {"label": "pos_07", "distance_mm": 700,  "angle_deg": 45},
]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tracker = AtracsysTracker()
    tracker.connect()
    print("Tracker connected.\n")

    meta_lines = ["label,distance_mm,angle_deg"]

    for pos in POSITIONS:
        label = pos["label"]
        print(f"─── {label}  (distance={pos['distance_mm']} mm, "
              f"angle={pos['angle_deg']} deg) ───")
        input("  Place the rigid body at this position, then press Enter to collect...")

        print(f"  Collecting {N_SAMPLES} samples for '{BODY_NAME}'...")
        samples = tracker.collect_samples(BODY_NAME, n=N_SAMPLES)

        out_path = DATA_DIR / f"{label}.csv"
        save_poses_csv(str(out_path), samples)
        print(f"  Saved → {out_path}\n")

        meta_lines.append(f"{label},{pos['distance_mm']},{pos['angle_deg']}")

    (DATA_DIR / "positions.txt").write_text("\n".join(meta_lines))
    tracker.disconnect()
    print("Done. All data saved to", DATA_DIR)


if __name__ == "__main__":
    main()
