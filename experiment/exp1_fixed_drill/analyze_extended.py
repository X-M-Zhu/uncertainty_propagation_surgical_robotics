# Author: X.M. Christine Zhu

"""
Experiment 1 — Extended analysis: rotation block + ellipsoid-shape metrics.

analyze.py validates the covariance propagation formula using a single
scalar (translational sigma of T_AB) and a full-6x6 Frobenius error. This
script reuses analyse_position() from analyze.py (no duplicated logic) and
adds, for each position:
  - rotation-block sigma comparison (predicted vs. empirical, deg)
  - translation-block sigma comparison (predicted vs. empirical, mm)
  - rotation-translation cross-block error (off-diagonal C[:3, 3:])
  - ellipsoid shape comparison: principal-sigma ratios and principal-axis
    alignment angles, for both the rotation and translation blocks

This shows not just that the average magnitude of predicted uncertainty
matches empirical, but that the *shape* (relative size + orientation of the
uncertainty ellipsoid) matches too.

Produces
--------
  results_fixed_drill/extended_report.txt
  results_fixed_drill/fig_ellipsoid_shape_match.png
"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve().parent
_EXP = _HERE.parent
sys.path.insert(0, str(_EXP))

from analyze import analyse_position, DATA_DIR, RESULTS_DIR
from utils.se3_stats import summary_stats
from utils.uncertainty_metrics import (
    ROT_BLOCK,
    TRANS_BLOCK,
    block_compare,
    cross_block_compare,
    ellipsoid_shape_compare,
)


def main():
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
    dist_list = []
    trans_ratio_axes, trans_angle_axes = [], []

    print(
        f"\n{'Label':<10} {'σ_rot_emp':>10} {'σ_rot_pred':>11} {'rot rel%':>9} "
        f"{'σ_trans_emp':>12} {'σ_trans_pred':>13} {'trans rel%':>10} {'cross rel%':>10}"
    )
    print("─" * 90)

    for pos in positions:
        pos_dir = DATA_DIR / pos["label"]
        if not (pos_dir / "bodyA.csv").exists():
            print(f"  {pos['label']}: data not found, skipping")
            continue

        res = analyse_position(pos_dir)
        C_pred = res["C_AB_pred"]
        C_emp = res["C_AB_emp"]
        sA = res["summary_A"]

        rot_cmp = block_compare(C_pred, C_emp, ROT_BLOCK)
        trans_cmp = block_compare(C_pred, C_emp, TRANS_BLOCK)
        cross_cmp = cross_block_compare(C_pred, C_emp)
        rot_shape = ellipsoid_shape_compare(C_pred, C_emp, ROT_BLOCK)
        trans_shape = ellipsoid_shape_compare(C_pred, C_emp, TRANS_BLOCK)

        sigma_rot_emp_deg = np.degrees(rot_cmp["sigma_emp"])
        sigma_rot_pred_deg = np.degrees(rot_cmp["sigma_pred"])
        sigma_trans_emp_mm = trans_cmp["sigma_emp"] * 1000.0
        sigma_trans_pred_mm = trans_cmp["sigma_pred"] * 1000.0

        print(
            f"{pos['label']:<10} {sigma_rot_emp_deg:>10.4f} {sigma_rot_pred_deg:>11.4f} "
            f"{rot_cmp['rel_error_pct']:>8.2f}% "
            f"{sigma_trans_emp_mm:>12.4f} {sigma_trans_pred_mm:>13.4f} "
            f"{trans_cmp['rel_error_pct']:>9.2f}% {cross_cmp['rel_error_pct']:>9.2f}%"
        )

        report_lines.append(
            f"{pos['label']} (dist={sA['distance_mm']:.0f} mm): "
            f"rot σ_emp={sigma_rot_emp_deg:.4f}deg σ_pred={sigma_rot_pred_deg:.4f}deg "
            f"rel_err={rot_cmp['rel_error_pct']:.2f}%  "
            f"trans σ_emp={sigma_trans_emp_mm:.4f}mm σ_pred={sigma_trans_pred_mm:.4f}mm "
            f"rel_err={trans_cmp['rel_error_pct']:.2f}%  "
            f"cross_rel_err={cross_cmp['rel_error_pct']:.2f}%  "
            f"rot_axis_angle_deg={np.array2string(rot_shape['axis_angle_deg'], precision=2)}  "
            f"trans_axis_angle_deg={np.array2string(trans_shape['axis_angle_deg'], precision=2)}  "
            f"trans_sigma_ratio={np.array2string(trans_shape['sigma_ratio'], precision=3)}"
        )

        dist_list.append(sA["distance_mm"])
        trans_ratio_axes.append(trans_shape["sigma_ratio"])
        trans_angle_axes.append(trans_shape["axis_angle_deg"])

    (RESULTS_DIR / "extended_report.txt").write_text("\n".join(report_lines))
    print(f"\nSaved → {RESULTS_DIR / 'extended_report.txt'}")

    if len(dist_list) >= 2:
        order = np.argsort(dist_list)
        dists = np.array(dist_list)[order]
        ratios = np.array(trans_ratio_axes)[order]   # (n_pos, 3)
        angles = np.array(trans_angle_axes)[order]   # (n_pos, 3)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        for k in range(3):
            ax1.plot(dists, ratios[:, k], "o-", label=f"axis {k+1}")
        ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax1.set_xlabel("Drill distance from tracker (mm)")
        ax1.set_ylabel("σ ratio (predicted / empirical)")
        ax1.set_title("Translation-block principal σ ratio")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for k in range(3):
            ax2.plot(dists, angles[:, k], "o-", label=f"axis {k+1}")
        ax2.set_xlabel("Drill distance from tracker (mm)")
        ax2.set_ylabel("Axis misalignment (deg)")
        ax2.set_title("Translation-block principal-axis alignment")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        out = RESULTS_DIR / "fig_ellipsoid_shape_match.png"
        fig.savefig(str(out), dpi=150)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
