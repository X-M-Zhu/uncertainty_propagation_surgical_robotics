# Author: X.M. Christine Zhu

"""
Experiment 2 — Extended analysis: rotation block + ellipsoid-shape metrics.

Mirrors exp1's analyze_extended.py. Imports analyse_config() from analyze.py
(no duplicated logic) and adds per-configuration:
  - rotation-block sigma comparison (predicted vs. empirical, deg)
  - translation-block sigma comparison (predicted vs. empirical, mm)
  - rotation-translation cross-block error
  - ellipsoid shape: principal-sigma ratios and principal-axis alignment angles

Produces
--------
  results/extended_report.txt
  results/fig_ellipsoid_shape_match.png
"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve().parent
_EXP  = _HERE.parent
sys.path.insert(0, str(_EXP))

from analyze import analyse_config, DATA_DIR, RESULTS_DIR
from utils.uncertainty_metrics import (
    ROT_BLOCK,
    TRANS_BLOCK,
    block_compare,
    cross_block_compare,
    ellipsoid_shape_compare,
)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg_dirs = sorted(DATA_DIR.glob("cfg_*"))
    if not cfg_dirs:
        raise FileNotFoundError(f"No cfg_* folders in {DATA_DIR}. Run collect.py first.")

    report_lines = []
    labels = []
    trans_ratio_axes, trans_angle_axes = [], []

    print(
        f"\n{'Config':<10} {'σ_rot_emp':>10} {'σ_rot_pred':>11} {'rot rel%':>9} "
        f"{'σ_trans_emp':>12} {'σ_trans_pred':>13} {'trans rel%':>10} {'cross rel%':>10}"
    )
    print("─" * 90)

    for cfg_dir in cfg_dirs:
        label = cfg_dir.name
        try:
            res = analyse_config(cfg_dir)
        except FileNotFoundError as e:
            print(f"  {label}: skipping — {e}")
            continue

        C_pred = res["C_AB_pred"]
        C_emp  = res["C_AB_emp"]

        rot_cmp   = block_compare(C_pred, C_emp, ROT_BLOCK)
        trans_cmp = block_compare(C_pred, C_emp, TRANS_BLOCK)
        cross_cmp = cross_block_compare(C_pred, C_emp)
        trans_shape = ellipsoid_shape_compare(C_pred, C_emp, TRANS_BLOCK)
        rot_shape   = ellipsoid_shape_compare(C_pred, C_emp, ROT_BLOCK)

        sigma_rot_emp_deg  = np.degrees(rot_cmp["sigma_emp"])
        sigma_rot_pred_deg = np.degrees(rot_cmp["sigma_pred"])
        sigma_trans_emp_mm  = trans_cmp["sigma_emp"]  * 1000.0
        sigma_trans_pred_mm = trans_cmp["sigma_pred"] * 1000.0

        print(
            f"{label:<10} {sigma_rot_emp_deg:>10.4f} {sigma_rot_pred_deg:>11.4f} "
            f"{rot_cmp['rel_error_pct']:>8.2f}% "
            f"{sigma_trans_emp_mm:>12.4f} {sigma_trans_pred_mm:>13.4f} "
            f"{trans_cmp['rel_error_pct']:>9.2f}% {cross_cmp['rel_error_pct']:>9.2f}%"
        )

        report_lines.append(
            f"{label}: "
            f"rot σ_emp={sigma_rot_emp_deg:.4f}deg σ_pred={sigma_rot_pred_deg:.4f}deg "
            f"rel_err={rot_cmp['rel_error_pct']:.2f}%  "
            f"trans σ_emp={sigma_trans_emp_mm:.4f}mm σ_pred={sigma_trans_pred_mm:.4f}mm "
            f"rel_err={trans_cmp['rel_error_pct']:.2f}%  "
            f"cross_rel_err={cross_cmp['rel_error_pct']:.2f}%  "
            f"trans_axis_angle_deg={np.array2string(trans_shape['axis_angle_deg'], precision=2)}  "
            f"trans_sigma_ratio={np.array2string(trans_shape['sigma_ratio'], precision=3)}"
        )

        labels.append(label)
        trans_ratio_axes.append(trans_shape["sigma_ratio"])
        trans_angle_axes.append(trans_shape["axis_angle_deg"])

    (RESULTS_DIR / "extended_report.txt").write_text("\n".join(report_lines))
    print(f"\nSaved → {RESULTS_DIR / 'extended_report.txt'}")

    if len(labels) >= 2:
        x      = np.arange(len(labels))
        ratios = np.array(trans_ratio_axes)   # (n_cfg, 3)
        angles = np.array(trans_angle_axes)   # (n_cfg, 3)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        for k in range(3):
            ax1.plot(x, ratios[:, k], "o-", label=f"axis {k+1}")
        ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=15)
        ax1.set_ylabel("σ ratio (predicted / empirical)")
        ax1.set_title("Translation-block principal σ ratio")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for k in range(3):
            ax2.plot(x, angles[:, k], "o-", label=f"axis {k+1}")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=15)
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
