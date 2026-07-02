# Author: X.M. Christine Zhu

"""
Experiment 1 — Analysis: Anatomy fixed, drill moves.

For each drill position this script shows:
  - C_TA  : empirical covariance of the drill in tracker frame (A = drill here)
  - C_AB  : empirical covariance of relative pose T_AB = inv(T_TA) @ T_TB
  - C_AB_pred : C_AB predicted by propagating C_TA through inv() (C_TB negligible)
  - Frobenius error between C_AB_pred and C_AB_empirical

Distance from the tracker to the drill is computed automatically from the
empirical mean pose (no manual measurement needed). The angle_note field is
a free-form label you optionally typed during collection — purely for your
own reference, not used numerically.

Key result
----------
As the drill moves further from the tracker, σ_TA grows.
The predicted σ_AB should grow in the same proportion — confirming that the
adjoint-based propagation formula captures how the drill's positional
uncertainty feeds into the relative transform uncertainty.

Produces
--------
  results_fixed_drill/fig_sigma_vs_drill_distance.png
  results_fixed_drill/per_position_report.txt
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

from utils.se3_stats import load_poses_csv, se3_empirical_stats, summary_stats
from uncertainty_networks import UncertainTransform
from uncertainty_networks.se3 import inv_se3

DATA_DIR    = _HERE / "data_fixed_drill"
RESULTS_DIR = _HERE / "results_fixed_drill"


def analyse_position(pos_dir: pathlib.Path) -> dict:
    samples_A = load_poses_csv(str(pos_dir / "bodyA.csv"))
    samples_B = load_poses_csv(str(pos_dir / "bodyB.csv"))
    n = min(len(samples_A), len(samples_B))
    samples_A, samples_B = samples_A[:n], samples_B[:n]

    # Empirical covariances
    mean_A, C_A = se3_empirical_stats(samples_A)   # C of T_TA (drill, moved)
    mean_B, C_B = se3_empirical_stats(samples_B)   # C of T_TB (Anatomy, fixed — should be small)

    # Empirical relative pose covariance from simultaneous pairs
    rel_samples  = np.array([inv_se3(samples_A[i]) @ samples_B[i] for i in range(n)])
    mean_AB, C_AB_emp = se3_empirical_stats(rel_samples)

    # Predicted relative pose covariance: propagate C_TA and C_TB through
    # T_AB = inv(T_TA) @ T_TB  using UncertainTransform.inv() and compose()
    U_TA     = UncertainTransform(mean_A,  C_A)
    U_TB     = UncertainTransform(mean_B,  C_B)
    U_TA_inv = U_TA.inv()                           # inv(T_TA) with propagated C
    U_AB_pred = U_TA_inv.compose(U_TB)              # inv(T_TA) @ T_TB

    frob_error = np.linalg.norm(U_AB_pred.C - C_AB_emp, "fro")
    rel_error  = frob_error / (np.linalg.norm(C_AB_emp, "fro") + 1e-30)

    return {
        "C_TA":       C_A,
        "C_TB":       C_B,
        "C_AB_emp":   C_AB_emp,
        "C_AB_pred":  U_AB_pred.C,
        "mean_AB":    mean_AB,
        "frob_error": frob_error,
        "rel_error_pct": rel_error * 100.0,
        "summary_A":  summary_stats(mean_A,  C_A),
        "summary_B":  summary_stats(mean_B,  C_B),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load position metadata (label, optional free-form angle_note)
    meta_path = DATA_DIR / "positions.txt"
    if not meta_path.exists():
        raise FileNotFoundError(f"Run collect.py first. Expected: {meta_path}")
    rows = meta_path.read_text().strip().split("\n")[1:]
    positions = []
    for row in rows:
        parts = row.split(",", 1)
        label = parts[0].strip()
        angle_note = parts[1].strip() if len(parts) > 1 else ""
        positions.append({"label": label, "angle_note": angle_note})

    # Per-position analysis
    records = []
    print(f"\n{'Label':<10} {'Dist(mm)':>9} {'Note':<14} "
          f"{'σ_TA_t(mm)':>12} {'σ_AB_emp(mm)':>14} {'σ_AB_pred(mm)':>15} "
          f"{'Frob err':>10} {'Rel err(%)':>11}")
    print("─" * 100)

    report_lines = []
    for pos in positions:
        pos_dir = DATA_DIR / pos["label"]
        if not (pos_dir / "bodyA.csv").exists():
            print(f"  {pos['label']}: data not found, skipping")
            continue

        res = analyse_position(pos_dir)
        sA  = res["summary_A"]    # distance_mm computed automatically from the data
        sAB_emp  = summary_stats(res["mean_AB"], res["C_AB_emp"])
        sAB_pred = summary_stats(res["mean_AB"], res["C_AB_pred"])

        print(f"{pos['label']:<10} {sA['distance_mm']:>9.0f} {pos['angle_note']:<14} "
              f"{sA['sigma_trans_mm']:>12.4f} "
              f"{sAB_emp['sigma_trans_mm']:>14.4f} "
              f"{sAB_pred['sigma_trans_mm']:>15.4f} "
              f"{res['frob_error']:>10.6f} "
              f"{res['rel_error_pct']:>10.2f}%")

        records.append({**pos, **res, "distance_mm": sA["distance_mm"]})
        report_lines.append(
            f"{pos['label']} (dist={sA['distance_mm']:.0f} mm, note='{pos['angle_note']}'): "
            f"σ_TA_trans={sA['sigma_trans_mm']:.4f} mm  "
            f"σ_AB_emp={sAB_emp['sigma_trans_mm']:.4f} mm  "
            f"σ_AB_pred={sAB_pred['sigma_trans_mm']:.4f} mm  "
            f"frob={res['frob_error']:.6f}  rel={res['rel_error_pct']:.2f}%"
        )

    # ── Plot: σ_TA and σ_AB vs drill distance (auto-computed, no angle grouping) ──
    if len(records) >= 2:
        order       = sorted(range(len(records)), key=lambda i: records[i]["distance_mm"])
        dists       = [records[i]["distance_mm"] for i in order]
        sA_trans    = [summary_stats(records[i]["mean_AB"], records[i]["C_TA"])["sigma_trans_mm"]
                       for i in order]
        sAB_emp_t   = [summary_stats(records[i]["mean_AB"], records[i]["C_AB_emp"])["sigma_trans_mm"]
                       for i in order]
        sAB_pred_t  = [summary_stats(records[i]["mean_AB"], records[i]["C_AB_pred"])["sigma_trans_mm"]
                       for i in order]

        diff_t = [p - e for p, e in zip(sAB_pred_t, sAB_emp_t)]

        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                      gridspec_kw={"height_ratios": [3, 1]},
                                      sharex=True)
        ax.plot(dists, sA_trans,   "o--", color="gray",      label="σ_TA (drill direct)")
        ax.plot(dists, sAB_emp_t,  "o-",  color="tomato",    label="σ_AB empirical")
        ax.plot(dists, sAB_pred_t, "s-",  color="steelblue", label="σ_AB predicted")
        ax.set_ylabel("σ translation (mm)")
        ax.set_title("Fixed-Anatomy experiment: relative pose uncertainty vs drill distance")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax2.bar(dists, diff_t, width=12, color="steelblue", alpha=0.7)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Drill distance from tracker (mm, auto-computed)")
        ax2.set_ylabel("pred − emp (mm)")
        ax2.set_title("Prediction error (σ_AB predicted − empirical)")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        out = RESULTS_DIR / "fig_sigma_vs_drill_distance.png"
        fig.savefig(str(out), dpi=150)
        print(f"\nSaved → {out}")

    # ── Save text report ─────────────────────────────────────────────────────
    rpt_path = RESULTS_DIR / "per_position_report.txt"
    rpt_path.write_text("\n".join(report_lines))
    print(f"Saved → {rpt_path}")
    plt.show()


if __name__ == "__main__":
    main()
