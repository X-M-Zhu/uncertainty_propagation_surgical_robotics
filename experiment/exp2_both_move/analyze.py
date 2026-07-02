# Author: X.M. Christine Zhu

"""
Experiment 2 — Analysis: both Anatomy and Drill move.

Key difference from exp1: both T_TA (Anatomy) and T_TB (Drill) have
significant uncertainty. We validate that the framework correctly combines
both sources when predicting C_AB.

Chain
-----
    Tracker --[T_TA]--> Anatomy --[T_AB]--> Drill

  T_TA  : pose of Anatomy in tracker frame      (bodyA.csv)
  T_TB  : pose of Drill   in tracker frame      (bodyB.csv)
  T_AB  : pose of Drill   in Anatomy frame      = inv(T_TA) @ T_TB

Validation
----------
  C_AB_pred  :  predicted by propagating C_TA and C_TB through inv/compose
  C_AB_emp   :  empirical from paired simultaneous samples inv(T_TA[i])@T_TB[i]

Tip point uncertainty
---------------------
  p_tip_local  loaded from experiment/shared_cal/tip_offset.json (run
               exp1_fixed_drill/collect_pivot.py + calibrate_pivot.py once).
  C_tip        3×3 covariance of the drill tip in the Anatomy frame,
               propagated through T_AB via the linearised Jacobian.

Produces
--------
  results/fig_sigma_comparison.png
  results/fig_frobenius_error.png
  results/summary_report.txt
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

DATA_DIR    = _HERE / "data"
RESULTS_DIR = _HERE / "results"


def analyse_config(cfg_dir: pathlib.Path, p_tip_local: np.ndarray = None) -> dict:
    samples_A = load_poses_csv(str(cfg_dir / "bodyA.csv"))   # Anatomy in tracker frame
    samples_B = load_poses_csv(str(cfg_dir / "bodyB.csv"))   # Drill   in tracker frame
    n = min(len(samples_A), len(samples_B))
    samples_A, samples_B = samples_A[:n], samples_B[:n]

    # Empirical stats for each body independently
    mean_A, C_A = se3_empirical_stats(samples_A)   # T_TA : Anatomy in tracker
    mean_B, C_B = se3_empirical_stats(samples_B)   # T_TB : Drill   in tracker

    # Empirical relative pose T_AB = inv(T_TA) @ T_TB (Drill in Anatomy frame)
    rel_samples      = np.array([inv_se3(samples_A[i]) @ samples_B[i] for i in range(n)])
    mean_AB, C_AB_emp = se3_empirical_stats(rel_samples)

    # Predicted C_AB: propagate C_A and C_B through inv(T_TA) @ T_TB
    U_A       = UncertainTransform(mean_A,  C_A)
    U_B       = UncertainTransform(mean_B,  C_B)
    U_AB_pred = U_A.inv().compose(U_B)

    frob_error = np.linalg.norm(U_AB_pred.C - C_AB_emp, "fro")
    rel_error  = frob_error / (np.linalg.norm(C_AB_emp, "fro") + 1e-30)

    result = {
        "mean_A":        mean_A,
        "mean_B":        mean_B,
        "mean_AB":       mean_AB,
        "C_A":           C_A,
        "C_B":           C_B,
        "C_AB_emp":      C_AB_emp,
        "C_AB_pred":     U_AB_pred.C,
        "frob_error":    frob_error,
        "rel_error_pct": rel_error * 100.0,
        "summary_A":     summary_stats(mean_A,  C_A),
        "summary_B":     summary_stats(mean_B,  C_B),
        "summary_AB_emp":  summary_stats(mean_AB, C_AB_emp),
        "summary_AB_pred": summary_stats(mean_AB, U_AB_pred.C),
    }

    if p_tip_local is not None:
        # Empirical tip positions in Anatomy frame from paired samples
        p_tips = np.array([s[:3, :3] @ p_tip_local + s[:3, 3] for s in rel_samples])
        result["C_tip_emp"]  = np.cov(p_tips.T)
        result["C_tip_pred"] = tip_point_covariance(mean_AB, U_AB_pred.C, p_tip_local)

    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load optional pivot calibration (shared across experiments)
    tip_json = _EXP / "shared_cal" / "tip_offset.json"
    p_tip_local = None
    if tip_json.exists():
        tip_data = json.loads(tip_json.read_text())
        p_tip_local = np.array([tip_data["x_mm"], tip_data["y_mm"],
                                 tip_data["z_mm"]]) * 1e-3
        print(f"Pivot calibration loaded: tip offset = {p_tip_local * 1000} mm  "
              f"(cal RMS = {tip_data['rms_residual_mm']:.3f} mm)")
    else:
        print(f"No pivot calibration at {tip_json} — tip uncertainty will not be reported.\n"
              f"Run exp1_fixed_drill/collect_pivot.py + calibrate_pivot.py to enable it.")

    cfg_dirs = sorted(DATA_DIR.glob("cfg_*"))
    if not cfg_dirs:
        raise FileNotFoundError(
            f"No cfg_* folders found in {DATA_DIR}. Run collect.py first.")

    has_tip = p_tip_local is not None
    records = []

    hdr = (f"\n{'Config':<10} {'σ_A(mm)':>9} {'σ_B(mm)':>9} "
           f"{'σ_AB_emp(mm)':>14} {'σ_AB_pred(mm)':>15} "
           f"{'Frob err':>10} {'Rel err(%)':>11}")
    if has_tip:
        hdr += f"  {'σ_tip_emp(mm)':>14} {'σ_tip_pred(mm)':>15}"
    print(hdr)
    print("─" * (82 + (32 if has_tip else 0)))

    report_lines = []
    for cfg_dir in cfg_dirs:
        label = cfg_dir.name
        try:
            res = analyse_config(cfg_dir, p_tip_local=p_tip_local)
        except FileNotFoundError as e:
            print(f"  {label}: skipping — {e}")
            continue

        sA   = res["summary_A"]
        sB   = res["summary_B"]
        sABe = res["summary_AB_emp"]
        sABp = res["summary_AB_pred"]

        row = (f"{label:<10} {sA['sigma_trans_mm']:>9.4f} {sB['sigma_trans_mm']:>9.4f} "
               f"{sABe['sigma_trans_mm']:>14.4f} {sABp['sigma_trans_mm']:>15.4f} "
               f"{res['frob_error']:>10.6f} {res['rel_error_pct']:>10.2f}%")
        rpt = (f"{label}: "
               f"σ_A={sA['sigma_trans_mm']:.4f} mm  "
               f"σ_B={sB['sigma_trans_mm']:.4f} mm  "
               f"σ_AB_emp={sABe['sigma_trans_mm']:.4f} mm  "
               f"σ_AB_pred={sABp['sigma_trans_mm']:.4f} mm  "
               f"frob={res['frob_error']:.6f}  rel={res['rel_error_pct']:.2f}%")

        if has_tip:
            sigma_tip_emp  = np.sqrt(np.trace(res["C_tip_emp"])  / 3.0) * 1000.0
            sigma_tip_pred = np.sqrt(np.trace(res["C_tip_pred"]) / 3.0) * 1000.0
            row += f"  {sigma_tip_emp:>14.4f} {sigma_tip_pred:>15.4f}"
            rpt += f"  σ_tip_emp={sigma_tip_emp:.4f} mm  σ_tip_pred={sigma_tip_pred:.4f} mm"
            res["sigma_tip_emp_mm"]  = sigma_tip_emp
            res["sigma_tip_pred_mm"] = sigma_tip_pred

        print(row)
        records.append({**res, "label": label})
        report_lines.append(rpt)

    (RESULTS_DIR / "summary_report.txt").write_text("\n".join(report_lines))
    print(f"\nSaved → {RESULTS_DIR / 'summary_report.txt'}")

    if not records:
        return

    labels    = [r["label"] for r in records]
    x         = np.arange(len(labels))
    sA_list   = [r["summary_A"]["sigma_trans_mm"]   for r in records]
    sB_list   = [r["summary_B"]["sigma_trans_mm"]   for r in records]
    sABe_list = [r["summary_AB_emp"]["sigma_trans_mm"]  for r in records]
    sABp_list = [r["summary_AB_pred"]["sigma_trans_mm"] for r in records]
    frob_list = [r["frob_error"]    for r in records]
    rel_list  = [r["rel_error_pct"] for r in records]

    # ── Figure 1: σ comparison across configurations ──────────────────────────
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 8),
                                  gridspec_kw={"height_ratios": [3, 1]},
                                  sharex=True)

    ax.plot(x, sA_list,   "o--", color="royalblue",  label="σ_A  Anatomy (tracker frame)")
    ax.plot(x, sB_list,   "s--", color="tomato",     label="σ_B  Drill   (tracker frame)")
    ax.plot(x, sABe_list, "o-",  color="darkorange",  label="σ_AB empirical  (Drill in Anatomy)")
    ax.plot(x, sABp_list, "s-",  color="forestgreen", label="σ_AB predicted")

    if has_tip:
        tip_e_list = [r["sigma_tip_emp_mm"]  for r in records]
        tip_p_list = [r["sigma_tip_pred_mm"] for r in records]
        ax.plot(x, tip_e_list, "^-",  color="purple",    label="σ_tip empirical  (tip in Anatomy)")
        ax.plot(x, tip_p_list, "^--", color="mediumpurple", label="σ_tip predicted")

    ax.set_ylabel("σ translation (mm)")
    ax.set_title("Exp 2 — Both bodies move: predicted vs. empirical uncertainty")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2.bar(x, rel_list, color="steelblue", alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel("Rel error (%)")
    ax2.set_title("C_AB relative Frobenius error  (predicted vs. empirical)")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out1 = RESULTS_DIR / "fig_sigma_comparison.png"
    fig.savefig(str(out1), dpi=150)
    print(f"Saved → {out1}")

    # ── Figure 2: predicted vs. empirical scatter (σ_AB) ─────────────────────
    fig2, ax3 = plt.subplots(figsize=(5, 5))
    ax3.scatter(sABe_list, sABp_list, zorder=3, color="steelblue", s=60)
    for i, lbl in enumerate(labels):
        ax3.annotate(lbl, (sABe_list[i], sABp_list[i]),
                     textcoords="offset points", xytext=(6, 3), fontsize=8)
    lim_lo = min(min(sABe_list), min(sABp_list)) * 0.9
    lim_hi = max(max(sABe_list), max(sABp_list)) * 1.1
    ax3.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1, label="ideal (pred=emp)")
    ax3.set_xlabel("σ_AB empirical (mm)")
    ax3.set_ylabel("σ_AB predicted (mm)")
    ax3.set_title("Predicted vs. empirical σ_AB\n(points on diagonal = perfect prediction)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = RESULTS_DIR / "fig_pred_vs_emp_scatter.png"
    fig2.savefig(str(out2), dpi=150)
    print(f"Saved → {out2}")

    plt.show()


if __name__ == "__main__":
    main()
