# Author: X.M. Christine Zhu

"""
Experiment 2 — Data collection: both Anatomy and Drill move together.

Goal
----
Verify that composing two uncertain transforms gives the same result as
directly measuring the composed transform.

Physical setup
--------------
    Tracker ──[F_AB]──> Body A ──[F_BC]──> Body B

Body A is a rigid body the tracker can see independently.
Body B is a rigid body physically attached to (or near) Body A so the tracker
can also see it independently.

Collect three sets of measurements at each configuration:
  1. BodyA alone  (gives F_AB empirically)
  2. BodyB alone  (gives F_TB empirically, where T = tracker)
  3. A pointer or separate rigid body at the "tip" of the chain
     — OR just use BodyB as the end of the chain

Then the composed frame is:  F_AB_composed = F_TA @ F_AB_relative
where F_AB_relative = inv(F_TA_mean) @ F_TB_mean  (computed once at the start).

Alternatively, if you can track all bodies simultaneously, do so.
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

# "Anatomy" is measured in the tracker frame.
# Both bodies are tracked independently in the tracker frame.
# "reference" is NOT set in managerMarker_test.json, so measured_cp() returns
# tracker-frame poses for both.  The relative transform T_AB is computed
# in Python as inv(T_A[i]) @ T_B[i] from simultaneous pairs (see analyze.py).
BODY_A = "Anatomy"       # tracker → Anatomy  (world link, tracker frame)
BODY_B = "Anspoch_drill" # tracker → Drill    (tracker frame)
N_SAMPLES   = 300   # measurements per body per configuration
N_CONFIGS   = 5     # number of different physical configurations to measure
DATA_DIR    = _HERE / "data"

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tracker = AtracsysTracker()
    tracker.connect()
    print("Tracker connected.\n")

    for cfg in range(1, N_CONFIGS + 1):
        cfg_label = f"cfg_{cfg:02d}"
        cfg_dir   = DATA_DIR / cfg_label
        cfg_dir.mkdir(exist_ok=True)

        print(f"═══ Configuration {cfg}/{N_CONFIGS} ({cfg_label}) ═══")
        input("  Place Anatomy + Drill in a new configuration, then press Enter...")

        # Collect SIMULTANEOUS pairs: one T_A and one T_B at each time step.
        # Pairing is required so relative poses inv(T_A[i]) @ T_B[i] are valid.
        samples_A, samples_B = [], []
        print(f"  Collecting {N_SAMPLES} simultaneous pairs...")
        attempts = 0
        while len(samples_A) < N_SAMPLES and attempts < N_SAMPLES * 3:
            attempts += 1
            try:
                T_A = tracker.get_pose(BODY_A)
                T_B = tracker.get_pose(BODY_B)
                samples_A.append(T_A)
                samples_B.append(T_B)
            except RuntimeError:
                pass   # one body briefly occluded — skip and retry

        if len(samples_A) < N_SAMPLES:
            print(f"  WARNING: only collected {len(samples_A)} pairs — check occlusion.")

        samples_A = np.stack(samples_A)
        samples_B = np.stack(samples_B)
        save_poses_csv(str(cfg_dir / "bodyA.csv"), samples_A)
        save_poses_csv(str(cfg_dir / "bodyB.csv"), samples_B)

        for tag, samples in [("Anatomy", samples_A), ("Drill", samples_B)]:
            _, C = se3_empirical_stats(samples)
            s_rot   = np.degrees(np.sqrt(np.trace(C[:3, :3]) / 3.0))
            s_trans = np.sqrt(np.trace(C[3:, 3:]) / 3.0) * 1000.0
            print(f"  {tag}: σ_rot={s_rot:.4f}°   σ_trans={s_trans:.4f} mm")

        print(f"  Saved → {cfg_dir}\n")

    tracker.disconnect()
    print("Done. All data saved to", DATA_DIR)


if __name__ == "__main__":
    main()
