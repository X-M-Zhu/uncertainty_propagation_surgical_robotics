# Author: X.M. Christine Zhu

"""
Experiment 1 — Task-level point uncertainty at the calibrated drill tip.

Uses the tip offset from calibrate_pivot.py and, for each already-collected
drill position, propagates the relative-pose covariance U_AB down to a single
3-D point (the tip) in the Anatomy frame.

Two quantities are compared at the tip, for each position:

Predicted
    U_AB.apply_to_point(p_tip_local) via the CIS-I right-perturbation
    Jacobian in UncertainTransform (uncertain_geometry.py:367).
    Gives a 3x3 predicted point covariance Cp_pred.

Empirical
    For every paired sample (T_TA_i, T_TB_i) already in the CSVs, compute
        T_AB_i = inv(T_TA_i) @ T_TB_i
        p_tip_i = T_AB_i[:3,:3] @ p_tip_local + T_AB_i[:3,3]
    Then compute empirical mean and covariance of the N tip positions via
    np.cov. This is a genuine point-scatter estimate, independent of the
    tangent-space C_AB already used in analyze.py.

Metrics per position
    RMS point uncertainty  = sqrt(trace(Cp)/3) * 1000   [mm]
    95 % confidence radius = sqrt(max_eigval) * chi2_ppf(0.95, df=3) [mm]
        (the longest semi-axis of the ellipsoid scaled to a 95% CI level)
    Frobenius error between Cp_pred and Cp_emp
    Relative error (%)

Prerequisite
-----------
Run collect_pivot.py then calibrate_pivot.py on the lab machine first.
This script uses data_fixed_drill/pivot_cal/tip_offset.json for the
calibrated drill-local tip offset.

Produces
--------
  results_fixed_drill/task_point_report.txt
  results_fixed_drill/fig_task_point_uncertainty_vs_distance.png
"""

import sys
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

_HERE = pathlib.Path(__file__).resolve().parent
_EXP = _HERE.parent
_ROOT = _EXP.parent
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(_ROOT / "src"))

from utils.se3_stats import load_poses_csv, se3_empirical_stats, summary_stats
from utils.uncertainty_metrics import ellipsoid_shape_compare, TRANS_BLOCK
from uncertainty_networks import UncertainTransform
from uncertainty_networks.se3 import inv_se3

DATA_DIR = _HERE / "data_fixed_drill"
PIVOT_DIR = DATA_DIR / "pivot_cal"
RESULTS_DIR = _HERE / "results_fixed_drill"

_CHI2_95_DOF3 = float(np.sqrt(chi2.ppf(0.95, df=3)))


def point_covariance_empirical(
    samples_A: np.ndarray, samples_B: np.ndarray, p_tip_local: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive empirical tip positions in the Anatomy frame per sample and return
    (mean_tip (3,), Cp_emp (3,3)).
    """
    n = min(len(samples_A), len(samples_B))
    tip_positions = np.zeros((n, 3))
    for i in range(n):
        T_AB = inv_se3(samples_A[i]) @ samples_B[i]
        tip_positions[i] = T_AB[:3, :3] @ p_tip_local + T_AB[:3, 3]
    mean_tip = tip_positions.mean(axis=0)
    Cp_emp = np.cov(tip_positions, rowvar=False)
    return mean_tip, Cp_emp


def point_metrics(Cp: np.ndarray) -> dict:
    eigvals = np.linalg.eigvalsh(Cp)
    eigvals = np.maximum(eigvals, 0.0)
    return {
        "rms_mm": float(np.sqrt(np.trace(Cp) / 3.0) * 1000.0),
        "r95_mm": float(np.sqrt(eigvals.max()) * _CHI2_95_DOF3 * 1000.0),
    }


def main():
    tip_path = PIVOT_DIR / "tip_offset.json"
    if not tip_path.exists():
        raise FileNotFoundError(
            f"Run collect_pivot.py then calibrate_pivot.py first. "
            f"Expected: {tip_path}"
        )
    tip_data = json.loads(tip_path.read_text())
    p_tip_local = np.array([tip_data["x_mm"], tip_data["y_mm"], tip_data["z_mm"]]) * 1e-3
    print(f"Loaded tip offset (m): {p_tip_local}")
    print(f"Calibration RMS residual: {tip_data.get('rms_residual_mm', 'n/a'):.4f} mm\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = DATA_DIR / "positions.txt"
    if not meta_path.exists():
        raise FileNotFoundError(f"Run collect.py first. Expected: {meta_path}")
    rows = meta_path.read_text().strip().split("\n")[1:]
    positions = []
    for row in rows:
        parts = row.split(",", 1)
        positions.append({"label": parts[0].strip()})

    report_lines = []
    dist_list, rms_pred_list, rms_emp_list = [], [], []
    r95_pred_list, r95_emp_list = [], []

    print(
        f"\n{'Label':<10} {'Dist(mm)':>9} "
        f"{'RMS_pred(mm)':>13} {'RMS_emp(mm)':>12} "
        f"{'r95_pred(mm)':>13} {'r95_emp(mm)':>12} "
        f"{'Frob err':>10} {'Rel err(%)':>11}"
    )
    print("─" * 100)

    for pos in positions:
        pos_dir = DATA_DIR / pos["label"]
        if not (pos_dir / "bodyA.csv").exists():
            print(f"  {pos['label']}: data not found, skipping")
            continue

        samples_A = load_poses_csv(str(pos_dir / "bodyA.csv"))
        samples_B = load_poses_csv(str(pos_dir / "bodyB.csv"))
        n = min(len(samples_A), len(samples_B))
        samples_A, samples_B = samples_A[:n], samples_B[:n]

        mean_A, C_A = se3_empirical_stats(samples_A)
        mean_B, C_B = se3_empirical_stats(samples_B)
        sA = summary_stats(mean_A, C_A)

        U_TA = UncertainTransform(mean_A, C_A)
        U_TB = UncertainTransform(mean_B, C_B)
        U_AB = U_TA.inv().compose(U_TB)
        _, Cp_pred = U_AB.apply_to_point(p_tip_local)

        _, Cp_emp = point_covariance_empirical(samples_A, samples_B, p_tip_local)

        m_pred = point_metrics(Cp_pred)
        m_emp = point_metrics(Cp_emp)
        frob_err = float(np.linalg.norm(Cp_pred - Cp_emp, "fro"))
        rel_err_pct = frob_err / (np.linalg.norm(Cp_emp, "fro") + 1e-30) * 100.0

        print(
            f"{pos['label']:<10} {sA['distance_mm']:>9.0f} "
            f"{m_pred['rms_mm']:>13.4f} {m_emp['rms_mm']:>12.4f} "
            f"{m_pred['r95_mm']:>13.4f} {m_emp['r95_mm']:>12.4f} "
            f"{frob_err:>10.6f} {rel_err_pct:>10.2f}%"
        )

        report_lines.append(
            f"{pos['label']} (dist={sA['distance_mm']:.0f} mm): "
            f"tip_RMS pred={m_pred['rms_mm']:.4f}mm emp={m_emp['rms_mm']:.4f}mm  "
            f"95%CI pred={m_pred['r95_mm']:.4f}mm emp={m_emp['r95_mm']:.4f}mm  "
            f"frob={frob_err:.6f} rel={rel_err_pct:.2f}%"
        )
        dist_list.append(sA["distance_mm"])
        rms_pred_list.append(m_pred["rms_mm"])
        rms_emp_list.append(m_emp["rms_mm"])
        r95_pred_list.append(m_pred["r95_mm"])
        r95_emp_list.append(m_emp["r95_mm"])

    (RESULTS_DIR / "task_point_report.txt").write_text("\n".join(report_lines))
    print(f"\nSaved → {RESULTS_DIR / 'task_point_report.txt'}")

    if len(dist_list) >= 2:
        order = np.argsort(dist_list)
        dists = np.array(dist_list)[order]
        rms_pred = np.array(rms_pred_list)[order]
        rms_emp = np.array(rms_emp_list)[order]
        r95_pred = np.array(r95_pred_list)[order]
        r95_emp = np.array(r95_emp_list)[order]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.plot(dists, rms_pred, "s-", color="steelblue", label="RMS predicted")
        ax1.plot(dists, rms_emp, "o-", color="tomato", label="RMS empirical")
        ax1.set_xlabel("Drill distance from tracker (mm)")
        ax1.set_ylabel("Tip position RMS uncertainty (mm)")
        ax1.set_title("Task-level tip RMS uncertainty vs drill distance")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(dists, r95_pred, "s-", color="steelblue", label="95% CI predicted")
        ax2.plot(dists, r95_emp, "o-", color="tomato", label="95% CI empirical")
        ax2.set_xlabel("Drill distance from tracker (mm)")
        ax2.set_ylabel("Tip position 95% confidence radius (mm)")
        ax2.set_title("Task-level tip 95% CI vs drill distance")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        out = RESULTS_DIR / "fig_task_point_uncertainty_vs_distance.png"
        fig.savefig(str(out), dpi=150)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
