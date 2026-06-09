# Author: X.M. Christine Zhu

"""
Experiment 1 — Analysis: Tracker characterization.

Reads the CSVs produced by collect.py and produces:
  - Per-position empirical mean and 6×6 covariance
  - Table of σ_rot (deg) and σ_trans (mm) vs. distance and angle
  - Two plots: σ vs. distance (fixed angle) and σ vs. angle (fixed distance)
  - results/tracker_covariances.npz  — all covariances for downstream use
"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve().parent
_EXP  = _HERE.parent
sys.path.insert(0, str(_EXP))

from utils.se3_stats import load_poses_csv, se3_empirical_stats, summary_stats

DATA_DIR    = _HERE / "data"
RESULTS_DIR = _HERE / "results"


def load_all():
    meta_path = DATA_DIR / "positions.txt"
    if not meta_path.exists():
        raise FileNotFoundError(f"Run collect.py first. Expected: {meta_path}")

    rows = meta_path.read_text().strip().split("\n")[1:]  # skip header
    positions = []
    for row in rows:
        label, dist, angle = row.split(",")
        positions.append({
            "label":       label.strip(),
            "distance_mm": float(dist),
            "angle_deg":   float(angle),
        })
    return positions


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    positions = load_all()

    records = []
    saved_covs = {}

    print(f"{'Label':<10} {'Dist(mm)':>10} {'Angle(°)':>10} "
          f"{'σ_rot(°)':>12} {'σ_trans(mm)':>14}")
    print("─" * 62)

    for pos in positions:
        csv_path = DATA_DIR / f"{pos['label']}.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping")
            continue

        samples = load_poses_csv(str(csv_path))
        mean_F, C = se3_empirical_stats(samples)
        s = summary_stats(mean_F, C)

        print(f"{pos['label']:<10} {pos['distance_mm']:>10.0f} "
              f"{pos['angle_deg']:>10.0f} "
              f"{s['sigma_rot_deg']:>12.5f} {s['sigma_trans_mm']:>14.5f}")

        records.append({**pos, **s, "C": C, "mean_F": mean_F})
        saved_covs[pos["label"]] = C

    # Save all covariances for downstream use in Exp 2 / Exp 3
    np.savez(str(RESULTS_DIR / "tracker_covariances.npz"), **saved_covs)

    # ── Plot σ vs. distance (angle ≈ 0) ──────────────────────────────────────
    dist_records = [r for r in records if r["angle_deg"] == 0]
    if len(dist_records) >= 2:
        dists  = [r["distance_mm"]  for r in dist_records]
        s_rot  = [r["sigma_rot_deg"]   for r in dist_records]
        s_trans = [r["sigma_trans_mm"] for r in dist_records]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(dists, s_rot,  "o-", color="steelblue")
        axes[0].set_xlabel("Distance from tracker (mm)")
        axes[0].set_ylabel("σ rotation (deg)")
        axes[0].set_title("Rotational noise vs. distance")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(dists, s_trans, "o-", color="tomato")
        axes[1].set_xlabel("Distance from tracker (mm)")
        axes[1].set_ylabel("σ translation (mm)")
        axes[1].set_title("Translational noise vs. distance")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(RESULTS_DIR / "fig1_sigma_vs_distance.png"), dpi=150)
        print(f"\nSaved → {RESULTS_DIR / 'fig1_sigma_vs_distance.png'}")

    # ── Plot σ vs. angle (distance ≈ 700 mm) ─────────────────────────────────
    angle_records = [r for r in records if abs(r["distance_mm"] - 700) < 50]
    if len(angle_records) >= 2:
        angles  = [r["angle_deg"]    for r in angle_records]
        s_rot   = [r["sigma_rot_deg"]   for r in angle_records]
        s_trans = [r["sigma_trans_mm"] for r in angle_records]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(angles, s_rot,   "o-", color="steelblue")
        axes[0].set_xlabel("Angle from optical axis (deg)")
        axes[0].set_ylabel("σ rotation (deg)")
        axes[0].set_title("Rotational noise vs. angle")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(angles, s_trans, "o-", color="tomato")
        axes[1].set_xlabel("Angle from optical axis (deg)")
        axes[1].set_ylabel("σ translation (mm)")
        axes[1].set_title("Translational noise vs. angle")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(RESULTS_DIR / "fig2_sigma_vs_angle.png"), dpi=150)
        print(f"Saved → {RESULTS_DIR / 'fig2_sigma_vs_angle.png'}")

    plt.show()


if __name__ == "__main__":
    main()
