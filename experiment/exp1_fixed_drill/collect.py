# Author: X.M. Christine Zhu

"""
Experiment 1 — Data collection: Anatomy fixed, drill moves.

Mentor's setup
--------------
The Anatomy rigid body is placed on the table and NEVER MOVED.
Only the Anspoch_drill body is moved to different positions.

What this tests
---------------
At each drill position the tracker noise C_TA (A = drill here) changes
(varies with distance and angle from the camera). The relative transform
T_AB = inv(T_TA) @ T_TB inherits that uncertainty. The question is: does our
propagation formula correctly predict the empirical spread in T_AB as the
drill moves around?

Because Anatomy is fixed, C_TB (direct measurement of Anatomy) stays
constant — it is just the tracker's intrinsic noise at that one location.
This gives a stable, independent ground-truth against which to compare.

Procedure
---------
1. Place Anatomy on the table.  DO NOT TOUCH IT for the entire session.
2. Move the drill to a new position, collect N_SAMPLES simultaneous pairs
   (drill, Anatomy), and repeat for as many positions as you like.
   No manual distance/angle measurement needed — actual tracker-to-drill
   distance is computed automatically from the data in analyze.py. You may
   optionally type a free-form note (e.g. "side", "far corner") per position
   purely for your own reference.
3. Type 'done' instead of moving to a new position when finished.
4. Run analyze.py to compare predicted vs empirical covariance at each
   drill position.

Data layout
-----------
data_fixed_drill/
  pos_01/  bodyA.csv   bodyB.csv
  pos_02/  bodyA.csv   bodyB.csv
  ...
  positions.txt   — label, angle_note per row (distance is auto-computed)
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

BODY_A = "Anspoch_drill"  # moved to different positions
BODY_B = "Anatomy"        # FIXED — do not move during the session

N_SAMPLES = 300           # simultaneous pairs per drill position

DATA_DIR = _HERE / "data_fixed_drill"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tracker = AtracsysTracker()
    tracker.connect()
    print("Tracker connected.\n")
    print("=" * 60)
    print("IMPORTANT: Place Anatomy on the table NOW.")
    print("Do NOT move it again until the session is complete.")
    print("=" * 60)
    input("\nPress Enter when Anatomy is placed and visible...")

    # Verify Anatomy is visible before starting
    try:
        T_anatomy_ref = tracker.get_pose(BODY_B)
        print(f"  Anatomy confirmed visible at t={T_anatomy_ref[:3, 3]} m\n")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        tracker.disconnect()
        return

    meta_lines = ["label,angle_note"]
    pos_index = 0

    while True:
        pos_index += 1
        label = f"pos_{pos_index:02d}"

        print(f"─── {label} ───")
        cmd = input(
            "  Move the drill to a new position, then press Enter "
            "(or type 'done' to finish): "
        ).strip().lower()
        if cmd == "done":
            break

        angle_note = input(
            "  Optional note about this position (e.g. 'side', 'far corner'), "
            "or press Enter to skip: "
        ).strip()

        pos_dir = DATA_DIR / label
        pos_dir.mkdir(exist_ok=True)

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

        # Quick per-position summary (distance computed automatically from data)
        _, C_A = se3_empirical_stats(samples_A)
        _, C_B = se3_empirical_stats(samples_B)
        sA_rot   = np.degrees(np.sqrt(np.trace(C_A[:3, :3]) / 3.0))
        sA_trans = np.sqrt(np.trace(C_A[3:, 3:]) / 3.0) * 1000.0
        sB_rot   = np.degrees(np.sqrt(np.trace(C_B[:3, :3]) / 3.0))
        sB_trans = np.sqrt(np.trace(C_B[3:, 3:]) / 3.0) * 1000.0
        dist_drill_mm = float(np.linalg.norm(samples_A[:, :3, 3].mean(axis=0))) * 1000.0
        print(f"  Drill (moved) : σ_rot={sA_rot:.4f}°  σ_trans={sA_trans:.4f} mm"
              f"  dist≈{dist_drill_mm:.0f} mm from tracker")
        print(f"  Anatomy (fixed): σ_rot={sB_rot:.4f}°  σ_trans={sB_trans:.4f} mm  (should be stable)\n")

        meta_lines.append(f"{label},{angle_note}")

    (DATA_DIR / "positions.txt").write_text("\n".join(meta_lines))
    tracker.disconnect()
    print("Done. All data saved to", DATA_DIR)
    print("Run analyze.py to compare predicted vs empirical covariance.")


if __name__ == "__main__":
    main()
