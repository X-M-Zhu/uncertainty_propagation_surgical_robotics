# Author: X.M. Christine Zhu

"""
Experiment 2 — Random error estimation via e_i(t) = F_ref(t)^{-1} · b_i(t).

Following Dr. R. H. Taylor's advice for factoring out camera motion when
estimating random error.

In exp2 both bodies move, so we use Body A (Anatomy) as the reference:

    e_i(t) = F_A(t)^{-1} · b_i(t)

where
    F_A(t)  = T_TA(t)  — Anatomy pose in tracker frame (bodyA.csv)
    b_i(t)  — drill marker i's position in the tracker frame, reprojected
               from T_TB(t) using the geometry JSON

This expresses the drill's markers in the Anatomy frame, cancelling any
rigid camera motion that shifts both F_A and b_i together.

Note: unlike exp1 where the Anatomy is fixed, here Body A is also moving.
The spread of e_i(t) therefore includes noise from BOTH bodies — it measures
the relative random noise between drill and anatomy, camera-motion-corrected,
not the drill's absolute noise alone.

Caveat (hardware limitation)
-----------------------------
b_i(t) is reprojected from the fitted pose T_TB(t), not a truly raw
pre-fit ball detection. See exp1/analyze_ei.py for full discussion.

Reads
-----
  data/cfg_*/bodyA.csv
  data/cfg_*/bodyB.csv
  hardware/atracsys/atracsys/core/share/geometry_anspoch_drill.json

Writes
------
  results/ei_report.txt
  results/fig_ei_sigma_vs_config.png
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
from uncertainty_networks.se3 import inv_se3

DATA_DIR    = _HERE / "data"
RESULTS_DIR = _HERE / "results"

GEOMETRY_DRILL = (
    _ROOT / "hardware" / "atracsys" / "atracsys" / "core" / "share"
    / "geometry_anspoch_drill.json"
)


def load_fiducials_m(geometry_path: pathlib.Path) -> np.ndarray:
    """Load fiducial local positions from geometry JSON, converted mm → m."""
    with open(geometry_path) as f:
        geom = json.load(f)
    return np.array(
        [[fid["x"], fid["y"], fid["z"]] for fid in geom["fiducials"]],
        dtype=np.float64,
    ) * 1e-3


def compute_ei(
    samples_A: np.ndarray,
    samples_B: np.ndarray,
    fid_drill_local: np.ndarray,
) -> np.ndarray:
    """
    Compute e_i(t) = F_A(t)^{-1} · b_i(t) for each drill marker i.

    Parameters
    ----------
    samples_A : (N, 4, 4)  Anatomy poses in tracker frame  (= F_ref)
    samples_B : (N, 4, 4)  Drill poses   in tracker frame
    fid_drill_local : (n_markers, 3)  drill marker local positions

    Returns
    -------
    ei : (N, n_markers, 3)
        Each drill marker expressed in the Anatomy frame, per sample.
        Equivalent to T_AB(t) applied to each local marker position.
    """
    N = min(len(samples_A), len(samples_B))
    n_markers = fid_drill_local.shape[0]
    ei = np.zeros((N, n_markers, 3), dtype=np.float64)

    for t in range(N):
        T_AB = inv_se3(samples_A[t]) @ samples_B[t]
        R_AB = T_AB[:3, :3]
        t_AB = T_AB[:3, 3]
        for i in range(n_markers):
            ei[t, i] = R_AB @ fid_drill_local[i] + t_AB

    return ei


def marker_sigma_mm(positions: np.ndarray) -> float:
    """Isotropic-equivalent sigma (mm) of a set of 3D positions (N, 3)."""
    C = np.cov(positions, rowvar=False)
    return float(np.sqrt(np.trace(C) / 3.0) * 1000.0)


def main():
    if not GEOMETRY_DRILL.exists():
        raise FileNotFoundError(f"Geometry file not found: {GEOMETRY_DRILL}")

    fid_drill_local = load_fiducials_m(GEOMETRY_DRILL)
    n_markers = len(fid_drill_local)
    print(f"Loaded {n_markers} drill markers.\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg_dirs = sorted(DATA_DIR.glob("cfg_*"))
    if not cfg_dirs:
        raise FileNotFoundError(f"No cfg_* folders in {DATA_DIR}. Run collect.py first.")

    print(
        f"\n{'Config':<10} "
        + "  ".join(f"{'m'+str(k)+' σ_ei(mm)':>12}" for k in range(n_markers))
        + f"  {'avg σ_ei(mm)':>13}"
    )
    print("─" * (10 + n_markers * 14 + 15))

    report_lines = []
    labels, avg_ei_list = [], []

    for cfg_dir in cfg_dirs:
        label = cfg_dir.name
        if not (cfg_dir / "bodyA.csv").exists():
            print(f"  {label}: data not found, skipping")
            continue

        samples_A = load_poses_csv(str(cfg_dir / "bodyA.csv"))
        samples_B = load_poses_csv(str(cfg_dir / "bodyB.csv"))

        ei = compute_ei(samples_A, samples_B, fid_drill_local)

        per_marker_sigma = [marker_sigma_mm(ei[:, k, :]) for k in range(n_markers)]
        avg_sigma = float(np.mean(per_marker_sigma))

        sigma_str = "  ".join(f"{s:>12.4f}" for s in per_marker_sigma)
        print(f"{label:<10}  {sigma_str}  {avg_sigma:>13.4f}")

        report_lines.append(
            f"{label}: "
            + " ".join(f"m{k}={per_marker_sigma[k]:.4f}mm" for k in range(n_markers))
            + f" avg={avg_sigma:.4f}mm"
        )
        labels.append(label)
        avg_ei_list.append(avg_sigma)

    (RESULTS_DIR / "ei_report.txt").write_text("\n".join(report_lines))
    print(f"\nSaved → {RESULTS_DIR / 'ei_report.txt'}")

    if len(labels) >= 2:
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, avg_ei_list, "o-", color="steelblue",
                label="avg σ (anatomy frame, camera-motion corrected)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel("Average per-marker σ (mm)")
        ax.set_title(
            "Exp 2 — Drill marker random error in Anatomy frame\n"
            r"$e_i(t) = F_A(t)^{-1} \cdot b_i(t)$  (camera motion factored out)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = RESULTS_DIR / "fig_ei_sigma_vs_config.png"
        fig.savefig(str(out), dpi=150)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
