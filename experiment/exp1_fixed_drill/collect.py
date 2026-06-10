# Author: X.M. Christine Zhu

"""
Experiment 1 — Data collection: drill fixed, only Anatomy moves.

Mentor's setup
--------------
The Anspoch_drill rigid body is placed on the table and NEVER MOVED.
Only the Anatomy body is moved to different positions.

What this tests
---------------
At each Anatomy position the tracker noise C_TA changes (varies with distance
and angle from the camera). The relative transform T_AB = inv(T_TA) @ T_TB
inherits that uncertainty. The question is: does our propagation formula
correctly predict the empirical spread in T_AB as Anatomy moves around?

Because the drill is fixed, C_TB (direct measurement of the drill) stays
constant — it is just the tracker's intrinsic noise at that one location.
This gives a stable, independent ground-truth against which to compare.

Procedure
---------
1. Place Anspoch_drill on the table.  DO NOT TOUCH IT for the entire session.
2. For each position in ANATOMY_POSITIONS, move the Anatomy body there and
   collect N_SAMPLES simultaneous pairs (Anatomy, Drill).
3. Run analyze.py (shared with collect.py results) to compare predicted vs
   empirical covariance at each Anatomy position.

Data layout
-----------
data_fixed_drill/
  pos_01/  bodyA.csv   bodyB.csv
  pos_02/  bodyA.csv   bodyB.csv
  ...
  positions.txt   — label, distance_mm, angle_deg per row
"""

import sys
import pathlib
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_EXP  = _HERE.parent
sys.path.insert(0, str(_EXP))

from utils.atracsys_interface import AtracsysTracker
from utils.se3_stats import save_poses_csv, se3_empirical_stats

# ── Configuration ─────────────────────────────────────────────────────────────

BODY_A = "Anatomy"        # moved to different positions
BODY_B = "Anspoch_drill"  # FIXED — do not move during the session

N_SAMPLES = 300           # simultaneous pairs per Anatomy position

# Describe the Anatomy positions you will physically use.
# Fill in approx distance (mm from tracker) and angle (deg from optical axis).
ANATOMY_POSITIONS = [
    {"label": "pos_01", "distance_mm": 500,  "angle_deg":  0},
    {"label": "pos_02", "distance_mm": 700,  "angle_deg":  0},
    {"label": "pos_03", "distance_mm": 900,  "angle_deg":  0},
    {"label": "pos_04", "distance_mm": 1100, "angle_deg":  0},
    {"label": "pos_05", "distance_mm": 700,  "angle_deg": 15},
    {"label": "pos_06", "distance_mm": 700,  "angle_deg": 30},
    {"label": "pos_07", "distance_mm": 700,  "angle_deg": 45},
]

DATA_DIR = _HERE / "data_fixed_drill"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tracker = AtracsysTracker()
    tracker.connect()
    print("Tracker connected.\n")
    print("=" * 60)
    print("IMPORTANT: Place the Anspoch_drill on the table NOW.")
    print("Do NOT move it again until the session is complete.")
    print("=" * 60)
    input("\nPress Enter when the drill is placed and visible...")

    # Verify drill is visible before starting
    try:
        T_drill_ref = tracker.get_pose(BODY_B)
        print(f"  Drill confirmed visible at t={T_drill_ref[:3, 3]} m\n")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        tracker.disconnect()
        return

    meta_lines = ["label,distance_mm,angle_deg"]

    for pos in ANATOMY_POSITIONS:
        label = pos["label"]
        pos_dir = DATA_DIR / label
        pos_dir.mkdir(exist_ok=True)

        print(f"─── {label}  (distance={pos['distance_mm']} mm, "
              f"angle={pos['angle_deg']} deg) ───")
        input(f"  Move Anatomy to this position, then press Enter...")

        # Collect simultaneous pairs
        samples_A, samples_B = [], []
        attempts = 0
        print(f"  Collecting {N_SAMPLES} simultaneous pairs...")
        while len(samples_A) < N_SAMPLES and attempts < N_SAMPLES * 3:
            attempts += 1
            try:
                T_A = tracker.get_pose(BODY_A)
                T_B = tracker.get_pose(BODY_B)
                samples_A.append(T_A)
                samples_B.append(T_B)
            except RuntimeError:
                pass   # brief occlusion — retry

        if len(samples_A) < N_SAMPLES:
            print(f"  WARNING: only {len(samples_A)} pairs collected.")

        samples_A = np.stack(samples_A)
        samples_B = np.stack(samples_B)
        save_poses_csv(str(pos_dir / "bodyA.csv"), samples_A)
        save_poses_csv(str(pos_dir / "bodyB.csv"), samples_B)

        # Quick per-position summary
        _, C_A = se3_empirical_stats(samples_A)
        _, C_B = se3_empirical_stats(samples_B)
        sA_rot   = np.degrees(np.sqrt(np.trace(C_A[:3, :3]) / 3.0))
        sA_trans = np.sqrt(np.trace(C_A[3:, 3:]) / 3.0) * 1000.0
        sB_rot   = np.degrees(np.sqrt(np.trace(C_B[:3, :3]) / 3.0))
        sB_trans = np.sqrt(np.trace(C_B[3:, 3:]) / 3.0) * 1000.0
        print(f"  Anatomy: σ_rot={sA_rot:.4f}°  σ_trans={sA_trans:.4f} mm")
        print(f"  Drill  : σ_rot={sB_rot:.4f}°  σ_trans={sB_trans:.4f} mm  (should be stable)\n")

        meta_lines.append(f"{label},{pos['distance_mm']},{pos['angle_deg']}")

    (DATA_DIR / "positions.txt").write_text("\n".join(meta_lines))
    tracker.disconnect()
    print("Done. All data saved to", DATA_DIR)
    print("Run analyze.py to compare predicted vs empirical covariance.")


if __name__ == "__main__":
    main()
