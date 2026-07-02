# Author: X.M. Christine Zhu

"""
Experiment 3 — Analysis: Estimate Galen uncertainty parameters.

Reads data from collect.py and estimates:
    sigma_static       — from sub-exp A (repeatability spread)
    sigma_base         — from sub-exp B (static spread at fixed config)
    sigma_total        — compare predicted vs. empirical at each workspace config

How each parameter is estimated
--------------------------------
sigma_base
    At each workspace config in sub-exp B, the robot is held perfectly still.
    Any spread in optical measurements = tracker noise + base registration error.
    We subtract out the tracker noise (from Exp 1 results) to get sigma_base.

sigma_static
    In sub-exp A the robot is commanded back to the same config 50 times.
    The spread = sigma_static (kinematic repeatability) + sigma_base.
    We subtract the sigma_base estimate to isolate sigma_static.

sigma_total check
    For each workspace config: use galen_fk() + sigma_total to predict the
    covariance of the tip. Compare to the empirical covariance from sub-exp B.

Results saved to results/sigma_estimates.txt
"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve().parent
_EXP  = _HERE.parent
_ROOT = _EXP.parent

sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "simulation"))

from utils.se3_stats import load_poses_csv, se3_empirical_stats, summary_stats
from uncertainty_networks import GeometricNetwork, UncertainTransform
from node_registry import NODES

DATA_DIR    = _HERE / "data"
RESULTS_DIR = _HERE / "results"

# Tracker noise from Experiment 1 (update this once you run Exp 1)
# These are σ_trans (m) and σ_rot (rad) of the Atracsys at typical operating distance
TRACKER_SIGMA_TRANS = 0.0003   # 0.3 mm — placeholder, replace with Exp 1 result
TRACKER_SIGMA_ROT   = 0.0005   # ~0.03 deg — placeholder, replace with Exp 1 result


def estimate_sigma_base() -> float:
    """
    Estimate sigma_base from sub-exp B (static measurements, robot held still).

    The empirical translational std across static samples =
        sqrt(sigma_tracker^2 + sigma_base^2)

    So:  sigma_base = sqrt(empirical_sigma_trans^2 - sigma_tracker^2)
    """
    ws_dir = DATA_DIR / "workspace"
    if not ws_dir.exists():
        raise FileNotFoundError(f"Run sub-exp B first. Expected: {ws_dir}")

    cfg_dirs = sorted(ws_dir.glob("config_*"))
    sigma_trans_values = []

    for cfg_dir in cfg_dirs:
        csv_path = cfg_dir / "tip_poses.csv"
        if not csv_path.exists():
            continue
        samples = load_poses_csv(str(csv_path))
        _, C = se3_empirical_stats(samples)
        sigma_trans = np.sqrt(np.trace(C[3:, 3:]) / 3.0)
        sigma_trans_values.append(sigma_trans)

    median_sigma = np.median(sigma_trans_values)
    sigma_base_sq = max(0.0, median_sigma**2 - TRACKER_SIGMA_TRANS**2)
    sigma_base = float(np.sqrt(sigma_base_sq))

    print(f"  Median empirical σ_trans at static configs: {median_sigma*1000:.4f} mm")
    print(f"  Tracker σ_trans (from Exp 1):               {TRACKER_SIGMA_TRANS*1000:.4f} mm")
    print(f"  Estimated σ_base:                           {sigma_base*1000:.4f} mm")
    return sigma_base


def estimate_sigma_static(sigma_base: float) -> float:
    """
    Estimate sigma_static from sub-exp A (repeatability test).

    The translational spread across returns to Q_STAR =
        sqrt(sigma_static^2 + sigma_base^2 + sigma_tracker^2)

    Solving for sigma_static.
    """
    rep_dir = DATA_DIR / "repeatability"
    poses_path = rep_dir / "tip_poses.csv"
    if not poses_path.exists():
        raise FileNotFoundError(f"Run sub-exp A first. Expected: {poses_path}")

    samples = load_poses_csv(str(poses_path))
    _, C = se3_empirical_stats(samples)
    sigma_trans_total = np.sqrt(np.trace(C[3:, 3:]) / 3.0)

    sigma_static_sq = max(0.0,
        sigma_trans_total**2 - sigma_base**2 - TRACKER_SIGMA_TRANS**2)
    sigma_static = float(np.sqrt(sigma_static_sq))

    print(f"\n  Repeatability σ_trans (total):  {sigma_trans_total*1000:.4f} mm")
    print(f"  σ_base (already estimated):     {sigma_base*1000:.4f} mm")
    print(f"  Tracker σ_trans:                {TRACKER_SIGMA_TRANS*1000:.4f} mm")
    print(f"  Estimated σ_static:             {sigma_static*1000:.4f} mm")
    return sigma_static


def validate_sigma_total(sigma_static: float, sigma_base: float):
    """
    Compare framework-predicted tip covariance vs. empirical across workspace configs.
    """
    ws_dir = DATA_DIR / "workspace"
    joints_path = ws_dir / "joint_angles.csv"
    if not joints_path.exists():
        return

    joint_data = np.loadtxt(str(joints_path), delimiter=",", skiprows=1)
    cfg_dirs   = sorted(ws_dir.glob("config_*"))

    encoder_res = NODES["Galen"]["encoder_resolution"]
    sigma_enc   = encoder_res / np.sqrt(12)
    sigma_total = np.sqrt(sigma_static**2 + sigma_enc**2)

    frob_errors = []
    print(f"\n{'Config':<12} {'σ_total_pred(mm)':>18} {'σ_total_emp(mm)':>17} {'Frob err':>10}")
    print("─" * 62)

    for i, cfg_dir in enumerate(cfg_dirs):
        if i >= len(joint_data):
            break
        joints = joint_data[i]

        # Empirical covariance from static measurements
        csv_path = cfg_dir / "tip_poses.csv"
        if not csv_path.exists():
            continue
        samples = load_poses_csv(str(csv_path))
        _, C_emp = se3_empirical_stats(samples)

        # Framework prediction
        net = GeometricNetwork()
        T_base = np.eye(4)  # assume Galen base at origin for comparison
        base_C = np.diag([sigma_base**2] * 6)
        net.add_edge("World", "Base", UncertainTransform(T_base, base_C))

        transforms = NODES["Galen"]["fk"](joints)
        labels     = NODES["Galen"]["link_labels"]
        link_C     = np.diag([sigma_total**2] * 6)

        prev_T    = np.eye(4)
        prev_node = "Base"
        for T_k, label in zip(transforms, labels):
            cur_node = f"Galen_{label}"
            T_step   = np.linalg.inv(prev_T) @ T_k
            net.add_edge(prev_node, cur_node, UncertainTransform(T_step, link_C))
            prev_T    = T_k
            prev_node = cur_node

        tip_node = f"Galen_{labels[-1]}"
        result   = net.query_frame("World", tip_node)
        C_pred   = result.transform.C

        sigma_pred = np.sqrt(np.trace(C_pred[3:, 3:]) / 3.0) * 1000.0
        sigma_emp  = np.sqrt(np.trace(C_emp[3:, 3:])  / 3.0) * 1000.0
        frob       = np.linalg.norm(C_pred - C_emp, "fro")
        frob_errors.append(frob)

        print(f"{cfg_dir.name:<12} {sigma_pred:>18.4f} {sigma_emp:>17.4f} {frob:>10.6f}")

    return frob_errors


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("═══ Estimating σ_base from workspace sweep (sub-exp B) ═══")
    sigma_base = estimate_sigma_base()

    print("\n═══ Estimating σ_static from repeatability test (sub-exp A) ═══")
    sigma_static = estimate_sigma_static(sigma_base)

    encoder_res = NODES["Galen"]["encoder_resolution"]
    sigma_enc   = encoder_res / np.sqrt(12)
    sigma_total = np.sqrt(sigma_static**2 + sigma_enc**2)

    print("\n═══ Final parameter estimates ═══")
    print(f"  sigma_static       = {sigma_static:.6f} rad/m")
    print(f"  encoder_resolution = {encoder_res:.6f} rad  (from spec)")
    print(f"  sigma_enc          = {sigma_enc:.6f} rad  (= encoder_res / √12)")
    print(f"  sigma_base         = {sigma_base:.6f} m")
    print(f"  sigma_total        = {sigma_total:.6f} rad/m  (static + enc combined)")

    print("\n═══ Validating sigma_total across workspace configs ═══")
    frob_errors = validate_sigma_total(sigma_static, sigma_base)

    # Save report
    rpt_path = RESULTS_DIR / "sigma_estimates.txt"
    with open(rpt_path, "w") as f:
        f.write("Galen EE uncertainty parameter estimates\n")
        f.write("=" * 45 + "\n")
        f.write(f"sigma_static       = {sigma_static:.6f}\n")
        f.write(f"encoder_resolution = {encoder_res:.6f}\n")
        f.write(f"sigma_enc          = {sigma_enc:.6f}\n")
        f.write(f"sigma_base         = {sigma_base:.6f}\n")
        f.write(f"sigma_total        = {sigma_total:.6f}\n\n")
        f.write("Update node_registry.py NODES['Galen'] with these values.\n")
    print(f"\nSaved estimates → {rpt_path}")

    if frob_errors:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(1, len(frob_errors)+1), frob_errors, color="steelblue", alpha=0.8)
        ax.set_xlabel("Workspace config index")
        ax.set_ylabel("Frobenius norm  ||C_pred − C_emp||")
        ax.set_title("Galen σ_total validation across workspace")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(RESULTS_DIR / "fig_sigma_validation.png"), dpi=150)
        plt.show()


if __name__ == "__main__":
    main()
