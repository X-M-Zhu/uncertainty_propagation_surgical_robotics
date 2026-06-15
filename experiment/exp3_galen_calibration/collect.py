# Author: X.M. Christine Zhu

"""
Experiment 3 — Data collection: Galen robot calibration.

Goal
----
Estimate the three uncertainty parameters for the Galen EE robot:
    sigma_static       — backlash + compliance (repeatability at same joint command)
    encoder_resolution — from spec: 0.00018 rad (verify or refine)
    sigma_base         — base registration uncertainty

Two sub-experiments are run:

  Sub-exp A — REPEATABILITY  (estimates sigma_static + sigma_base)
    Command Galen to a fixed configuration q*.
    Move Galen away, then back to q*, 50 times.
    Measure tip pose with optical tracker each time.
    The spread = sigma_static (kinematic) + sigma_base (registration).

  Sub-exp B — WORKSPACE SWEEP  (validates sigma_total across configs)
    Command Galen to N_CONFIGS different configurations.
    At each config, collect 50 measurements without moving the robot.
    The spread at each config ≈ tracker noise + sigma_base.
    This lets you separate sigma_base from sigma_static in sub-exp A.

Data saved
----------
  data/repeatability/config_XX/tip.csv    (50 × 16 CSV)
  data/repeatability/joint_angles.csv     (50 × 5 CSV, one row per return)
  data/workspace/config_XX/tip.csv        (50 × 16 CSV)
  data/workspace/joint_angles.csv         (N_CONFIGS × 5 CSV)
"""

import sys
import pathlib
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_EXP  = _HERE.parent
_ROOT = _EXP.parent
sys.path.insert(0, str(_EXP))

from utils.atracsys_interface import AtracsysTracker
from utils.se3_stats import save_poses_csv

# ── Configuration ─────────────────────────────────────────────────────────────

TIP_BODY    = "Anspoch_drill"  # attached to Galen EE; pose returned in tracker frame
N_REPEAT    = 50           # how many times to return to q_star (sub-exp A)
N_CONFIGS   = 15           # workspace configurations (sub-exp B)
N_STATIC    = 50           # static measurements per config (sub-exp B)

# The repeatability target configuration (radians / meters for c1,c2,c3)
Q_STAR = np.array([0.0, 0.0, 0.0, 0.0, 0.0])   # all joints at zero

DATA_DIR = _HERE / "data"


# ── Galen controller interface ─────────────────────────────────────────────────
# Fill in your robot controller calls below.

def move_galen_to(joint_angles: np.ndarray) -> None:
    """
    Command Galen to the given joint angles and wait until motion stops.
    joint_angles: array of 5 floats  [c1, c2, c3, roll, tilt]
    """
    # TODO: replace with your Galen controller API
    # e.g.:  galen_controller.move_joints(joint_angles)
    #        galen_controller.wait_for_motion()
    input(f"  [MANUAL] Move Galen to joints {np.round(joint_angles, 4)}, then press Enter...")


def read_galen_joints() -> np.ndarray:
    """
    Read the current joint encoder values from Galen.
    Returns array of 5 floats  [c1, c2, c3, roll, tilt]
    """
    # TODO: replace with your Galen controller API
    # e.g.:  return np.array(galen_controller.get_joint_positions())
    raw = input("  [MANUAL] Enter current joint angles (5 values, space-separated): ")
    return np.array([float(v) for v in raw.split()])


# ── Sub-experiment A: Repeatability ──────────────────────────────────────────

def run_repeatability(tracker: AtracsysTracker):
    print("\n══════════════════════════════════════")
    print("Sub-experiment A: Repeatability test")
    print(f"  Target config: {Q_STAR}")
    print(f"  Repeats: {N_REPEAT}")
    print("══════════════════════════════════════\n")

    rep_dir = DATA_DIR / "repeatability"
    rep_dir.mkdir(parents=True, exist_ok=True)

    all_joints = []
    all_poses  = []

    for i in range(N_REPEAT):
        print(f"  Repeat {i+1}/{N_REPEAT}")

        # Move away (to a perturbed config) then back to Q_STAR
        perturb = Q_STAR + np.random.uniform(-0.05, 0.05, size=5)
        move_galen_to(perturb)
        move_galen_to(Q_STAR)

        joints = read_galen_joints()
        all_joints.append(joints)

        T = tracker.get_pose(TIP_BODY)
        all_poses.append(T)

    all_joints = np.array(all_joints)
    all_poses  = np.array(all_poses)

    np.savetxt(str(rep_dir / "joint_angles.csv"), all_joints, delimiter=",",
               header="c1,c2,c3,roll,tilt", comments="")
    save_poses_csv(str(rep_dir / "tip_poses.csv"), all_poses)
    print(f"  Saved repeatability data to {rep_dir}\n")


# ── Sub-experiment B: Workspace sweep ────────────────────────────────────────

def run_workspace_sweep(tracker: AtracsysTracker):
    print("══════════════════════════════════════")
    print("Sub-experiment B: Workspace sweep")
    print(f"  Configurations: {N_CONFIGS}  |  Static samples each: {N_STATIC}")
    print("══════════════════════════════════════\n")

    ws_dir = DATA_DIR / "workspace"
    ws_dir.mkdir(parents=True, exist_ok=True)

    workspace_configs = _sample_workspace_configs(N_CONFIGS)
    all_cmd_joints = []

    for idx, q in enumerate(workspace_configs):
        cfg_label = f"config_{idx+1:02d}"
        cfg_dir   = ws_dir / cfg_label
        cfg_dir.mkdir(exist_ok=True)

        print(f"  Config {idx+1}/{N_CONFIGS}: {np.round(q, 4)}")
        move_galen_to(q)
        joints = read_galen_joints()
        all_cmd_joints.append(joints)

        print(f"    Collecting {N_STATIC} static measurements...")
        samples = tracker.collect_samples(TIP_BODY, n=N_STATIC, verbose=False)
        save_poses_csv(str(cfg_dir / "tip_poses.csv"), samples)

    np.savetxt(str(ws_dir / "joint_angles.csv"),
               np.array(all_cmd_joints), delimiter=",",
               header="c1,c2,c3,roll,tilt", comments="")
    print(f"\n  Saved workspace sweep data to {ws_dir}")


def _sample_workspace_configs(n: int) -> list:
    """Generate n joint configurations spanning the Galen workspace."""
    # Galen joint ranges from node_registry NODES entry
    ranges = [
        (-0.02, 0.02),    # c1 (m)
        (-0.02, 0.02),    # c2 (m)
        (-0.02, 0.02),    # c3 (m)
        (-0.5,  0.5),     # roll (rad)
        (-0.3,  0.3),     # tilt (rad)
    ]
    configs = []
    for _ in range(n):
        q = np.array([np.random.uniform(lo, hi) for lo, hi in ranges])
        configs.append(q)
    return configs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tracker = AtracsysTracker()
    tracker.connect()
    print("Tracker connected.\n")

    choice = input("Run (A) repeatability, (B) workspace sweep, or (both)? [A/B/both]: ").strip().lower()

    if choice in ("a", "both"):
        run_repeatability(tracker)

    if choice in ("b", "both"):
        run_workspace_sweep(tracker)

    tracker.disconnect()
    print("\nAll data saved to", DATA_DIR)


if __name__ == "__main__":
    main()
