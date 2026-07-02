# Author: X.M. Christine Zhu

"""
Experiment 1 — Pivot calibration data collection: drill tip offset.

Determines the drill's local tip offset (tip position in the drill's own
tracked frame) so task-level (point) uncertainty can be propagated to the
actual tip, not just the registered rigid-body frame. Needed because both
geometry JSONs ship with pivot = {0,0,0} (uncalibrated).

Procedure
---------
1. Rest the drill tip in a fixed divot / pointed indentation (anything that
   keeps the tip's physical 3D position fixed in space).
2. Rotate the drill through as many different orientations as practical
   while keeping the tip planted in the divot — the more angular spread,
   the better-conditioned the calibration.
3. Press Enter to start collecting; the script gathers N_SAMPLES poses.
4. Run calibrate_pivot.py afterward to solve for the local tip offset.

Data layout
-----------
data_fixed_drill/pivot_cal/bodyA_pivot.csv
"""

import sys
import pathlib
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_EXP = _HERE.parent
sys.path.insert(0, str(_EXP))

from utils.atracsys_interface import AtracsysTracker
from utils.se3_stats import save_poses_csv

BODY_A = "Anspoch_drill"

N_SAMPLES = 300

DATA_DIR = _HERE / "data_fixed_drill"
PIVOT_DIR = DATA_DIR / "pivot_cal"


def main():
    PIVOT_DIR.mkdir(parents=True, exist_ok=True)

    tracker = AtracsysTracker()
    tracker.connect()
    print("Tracker connected.\n")
    print("=" * 60)
    print("Rest the drill tip in a fixed divot/point.")
    print("Rotate the drill through varied orientations while the tip")
    print("stays planted in that same physical point.")
    print("=" * 60)
    input("\nPress Enter to start collecting once the tip is planted...")

    samples_A = []
    attempts = 0
    print(f"Collecting {N_SAMPLES} samples while you rotate the drill...")
    while len(samples_A) < N_SAMPLES and attempts < N_SAMPLES * 3:
        attempts += 1
        try:
            T_A = tracker.get_pose(BODY_A)
            samples_A.append(T_A)
        except RuntimeError:
            pass  # brief occlusion — retry

    if len(samples_A) < N_SAMPLES:
        print(f"WARNING: only {len(samples_A)} samples collected.")

    samples_A = np.stack(samples_A)
    save_poses_csv(str(PIVOT_DIR / "bodyA_pivot.csv"), samples_A)

    tracker.disconnect()
    print(f"\nDone. Saved {len(samples_A)} samples to "
          f"{PIVOT_DIR / 'bodyA_pivot.csv'}")
    print("Run calibrate_pivot.py to solve for the local tip offset.")


if __name__ == "__main__":
    main()
