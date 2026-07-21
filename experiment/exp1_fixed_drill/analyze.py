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
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve().parent
_EXP  = _HERE.parent
_ROOT = _EXP.parent
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(_ROOT / "src"))

from utils.se3_stats import load_poses_csv, se3_empirical_stats, summary_stats
from utils.uncertainty_metrics import tip_point_covariance
from uncertainty_networks import UncertainTransform
from uncertainty_networks.se3 import inv_se3

DATA_DIR    = _HERE / "data_fixed_drill"
RESULTS_DIR = _HERE / "results_fixed_drill"


def analyse_position(pos_dir: pathlib.Path, p_tip_local: np.ndarray = None) -> dict:
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

    result = {
        "C_TA":          C_A,
        "C_TB":          C_B,
        "C_AB_emp":      C_AB_emp,
        "C_AB_pred":     U_AB_pred.C,
        "mean_A":        mean_A,
        "mean_AB":       mean_AB,
        "frob_error":    frob_error,
        "rel_error_pct": rel_error * 100.0,
        "summary_A":     summary_stats(mean_A, C_A),
        "summary_B":     summary_stats(mean_B, C_B),
    }

    if p_tip_local is not None:
        # Empirical 3D tip positions in tracker frame from individual drill samples
        p_tips = np.array([s[:3, :3] @ p_tip_local + s[:3, 3] for s in samples_A])
        result["C_tip_emp"]  = np.cov(p_tips.T)
        result["C_tip_pred"] = tip_point_covariance(mean_A, C_A, p_tip_local)

    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load optional pivot calibration result (from calibrate_pivot.py)
    tip_json = _EXP / "shared_cal" / "tip_offset.json"
    p_tip_local = None
    if tip_json.exists():
        tip_data = json.loads(tip_json.read_text())
        p_tip_local = np.array([tip_data["x_mm"], tip_data["y_mm"],
                                 tip_data["z_mm"]]) * 1e-3   # convert mm → m
        print(f"Pivot calibration loaded: tip offset = {p_tip_local * 1000} mm  "
              f"(cal RMS = {tip_data['rms_residual_mm']:.3f} mm)")
    else:
        print(f"No pivot calibration at {tip_json} — tip uncertainty will not be reported.\n"
              f"Run collect_pivot.py then calibrate_pivot.py to enable it.")

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
    has_tip = p_tip_local is not None
    hdr = (f"\n{'Label':<10} {'Dist(mm)':>9} {'Note':<14} "
           f"{'σ_TA_t(mm)':>12} {'σ_AB_emp(mm)':>14} {'σ_AB_pred(mm)':>15} "
           f"{'Frob err':>10} {'Rel err(%)':>11}")
    if has_tip:
        hdr += f"  {'σ_tip_emp(mm)':>14} {'σ_tip_pred(mm)':>15}"
    print(hdr)
    print("─" * (100 + (32 if has_tip else 0)))

    report_lines = []
    for pos in positions:
        pos_dir = DATA_DIR / pos["label"]
        if not (pos_dir / "bodyA.csv").exists():
            print(f"  {pos['label']}: data not found, skipping")
            continue

        res = analyse_position(pos_dir, p_tip_local=p_tip_local)
        sA       = res["summary_A"]
        sAB_emp  = summary_stats(res["mean_AB"], res["C_AB_emp"])
        sAB_pred = summary_stats(res["mean_AB"], res["C_AB_pred"])

        row = (f"{pos['label']:<10} {sA['distance_mm']:>9.0f} {pos['angle_note']:<14} "
               f"{sA['sigma_trans_mm']:>12.4f} "
               f"{sAB_emp['sigma_trans_mm']:>14.4f} "
               f"{sAB_pred['sigma_trans_mm']:>15.4f} "
               f"{res['frob_error']:>10.6f} "
               f"{res['rel_error_pct']:>10.2f}%")
        rpt = (f"{pos['label']} (dist={sA['distance_mm']:.0f} mm, note='{pos['angle_note']}'): "
               f"σ_TA_trans={sA['sigma_trans_mm']:.4f} mm  "
               f"σ_AB_emp={sAB_emp['sigma_trans_mm']:.4f} mm  "
               f"σ_AB_pred={sAB_pred['sigma_trans_mm']:.4f} mm  "
               f"frob={res['frob_error']:.6f}  rel={res['rel_error_pct']:.2f}%")

        if has_tip:
            sigma_tip_emp  = np.sqrt(np.trace(res["C_tip_emp"])  / 3.0) * 1000.0
            sigma_tip_pred = np.sqrt(np.trace(res["C_tip_pred"]) / 3.0) * 1000.0
            row += f"  {sigma_tip_emp:>14.4f} {sigma_tip_pred:>15.4f}"
            rpt += f"  σ_tip_emp={sigma_tip_emp:.4f} mm  σ_tip_pred={sigma_tip_pred:.4f} mm"
            res["sigma_tip_emp_mm"]  = sigma_tip_emp
            res["sigma_tip_pred_mm"] = sigma_tip_pred

        print(row)
        records.append({**pos, **res, "distance_mm": sA["distance_mm"]})
        report_lines.append(rpt)

    # ── Plot: σ_TA, σ_AB, and (optionally) σ_tip vs drill distance ───────────
    if len(records) >= 2:
        order      = sorted(range(len(records)), key=lambda i: records[i]["distance_mm"])
        dists      = [records[i]["distance_mm"] for i in order]
        sA_trans   = [summary_stats(records[i]["mean_AB"], records[i]["C_TA"])["sigma_trans_mm"]
                      for i in order]
        sAB_emp_t  = [summary_stats(records[i]["mean_AB"], records[i]["C_AB_emp"])["sigma_trans_mm"]
                      for i in order]
        sAB_pred_t = [summary_stats(records[i]["mean_AB"], records[i]["C_AB_pred"])["sigma_trans_mm"]
                      for i in order]
        diff_t     = [p - e for p, e in zip(sAB_pred_t, sAB_emp_t)]

        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                      gridspec_kw={"height_ratios": [3, 1]},
                                      sharex=True)
        ax.plot(dists, sA_trans,   "o--", color="gray",      label="σ_TA (drill direct)")
        ax.plot(dists, sAB_emp_t,  "o-",  color="tomato",    label="σ_AB empirical")
        ax.plot(dists, sAB_pred_t, "s-",  color="steelblue", label="σ_AB predicted")

        if has_tip:
            tip_emp_list  = [records[i]["sigma_tip_emp_mm"]  for i in order]
            tip_pred_list = [records[i]["sigma_tip_pred_mm"] for i in order]
            ax.plot(dists, tip_emp_list,  "^-",  color="darkorange",  label="σ_tip empirical")
            ax.plot(dists, tip_pred_list, "^--", color="forestgreen", label="σ_tip predicted")

        ax.set_yscale("log")
        ax.set_ylabel("σ translation (mm)  [log scale]")
        ax.set_title("Fixed-Anatomy experiment: pose & tip uncertainty vs drill distance")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")

        ax2.bar(dists, diff_t, width=12, color="steelblue", alpha=0.7)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Drill distance from tracker (mm, auto-computed)")
        ax2.set_ylabel("pred − emp (mm)")
        ax2.set_title("Prediction error  σ_AB (predicted − empirical)")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        out = RESULTS_DIR / "fig_sigma_vs_drill_distance.png"
        fig.savefig(str(out), dpi=150)
        print(f"\nSaved → {out}")

    # ── Save text report ──────────────────────────────────────────────────────
    rpt_path = RESULTS_DIR / "per_position_report.txt"
    rpt_path.write_text("\n".join(report_lines))
    print(f"Saved → {rpt_path}")
    plt.show()


if __name__ == "__main__":
    main()
