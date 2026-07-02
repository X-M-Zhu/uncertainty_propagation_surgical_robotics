# Author: X.M. Christine Zhu

"""
Experiment 1 — Random error estimation via e_i(t) = F_ref(t)^{-1} · b_i(t).

Following Dr. R. H. Taylor's advice for separating random from systematic error.

The idea
--------
A naive approach to estimating random error — watching how the Anatomy body's
measured mean drifts over time — is contaminated by *camera motion*: if the
tracker camera itself moves, all bodies shift together, making the reference
body appear to move even when it hasn't.

The remedy is to express each drill marker ball's tracker-frame position
*relative to the reference body (Anatomy)*:

    e_i(t) = F_ref(t)^{-1} · b_i(t)

where
    F_ref(t)  = T_TB(t)  — Anatomy pose in the tracker frame at sample t
    b_i(t)    — drill marker i's position in the tracker frame at sample t

Any rigid camera motion shifts F_ref and b_i identically, so it cancels in
the ratio. The residual spread of e_i(t) over N samples is a cleaner estimate
of the *random* error — tracker noise relative to the reference body, with
camera motion factored out.

Caveat (hardware limitation)
-----------------------------
b_i(t) is computed by reprojecting the drill's calibrated local marker
position through the fitted pose T_TA(t):

    b_i(t) = T_TA(t)[:3,:3] @ p_drill_local_i + T_TA(t)[:3,3]

It is NOT a truly raw, pre-fit ball position — the Atracsys SDK performs an
internal rigid-body fit before exposing any pose. As a consequence:
  - e_i(t) = inv(T_TB(t)) @ T_TA(t) @ p_local_i
           = T_AB(t) @ p_local_i
    i.e. the drill marker expressed in the anatomy frame via the relative pose.
  - Anatomy's own markers e_j(t) = p_anatomy_local_j (constant by construction,
    zero variance) — they are omitted from this analysis for that reason.

What this gives you
-------------------
The spread of e_i(t) is the camera-motion-corrected random noise of each drill
marker's position *in the anatomy frame*. Comparing this to the tracker-frame
spread (markersA.csv) shows how much camera motion inflated the raw estimate.

Reads
-----
  data_fixed_drill/<pos>/bodyA.csv
  data_fixed_drill/<pos>/bodyB.csv
  hardware/atracsys/atracsys/geometry_anspoch_drill.json

Writes
------
  results_fixed_drill/ei_report.txt
  results_fixed_drill/fig_ei_sigma_vs_distance.png
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

from utils.se3_stats import load_poses_csv, summary_stats, se3_empirical_stats
from uncertainty_networks.se3 import inv_se3

DATA_DIR    = _HERE / "data_fixed_drill"
RESULTS_DIR = _HERE / "results_fixed_drill"
GEOMETRY_DRILL = (
    _ROOT / "hardware" / "atracsys" / "atracsys" / "core" / "share" / "geometry_anspoch_drill.json"
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
    Compute e_i(t) = F_ref(t)^{-1} · b_i(t) for each drill marker i.

    Parameters
    ----------
    samples_A : (N, 4, 4)  drill poses in tracker frame
    samples_B : (N, 4, 4)  anatomy poses in tracker frame  (= F_ref)
    fid_drill_local : (n_markers, 3)  drill marker positions in drill frame

    Returns
    -------
    ei : (N, n_markers, 3)
        Each drill marker's position in the anatomy frame, per sample.
        Equivalent to T_AB(t) applied to each local marker position.
    """
    N = min(len(samples_A), len(samples_B))
    n_markers = fid_drill_local.shape[0]
    ei = np.zeros((N, n_markers, 3), dtype=np.float64)

    for t in range(N):
        # Relative pose: drill expressed in anatomy frame
        T_AB = inv_se3(samples_B[t]) @ samples_A[t]
        R_AB = T_AB[:3, :3]
        t_AB = T_AB[:3, 3]
        for i in range(n_markers):
            ei[t, i] = R_AB @ fid_drill_local[i] + t_AB

    return ei


def marker_sigma_mm(positions: np.ndarray) -> float:
    """
    Isotropic-equivalent sigma (mm) of a set of 3D positions.

    Parameters
    ----------
    positions : (N, 3)
    """
    C = np.cov(positions, rowvar=False)
    return float(np.sqrt(np.trace(C) / 3.0) * 1000.0)


def main():
    if not GEOMETRY_DRILL.exists():
        raise FileNotFoundError(f"Geometry file not found: {GEOMETRY_DRILL}")

    fid_drill_local = load_fiducials_m(GEOMETRY_DRILL)
    n_markers = len(fid_drill_local)
    print(f"Loaded {n_markers} drill markers.\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = DATA_DIR / "positions.txt"
    if not meta_path.exists():
        raise FileNotFoundError(f"Run collect.py first. Expected: {meta_path}")
    rows = meta_path.read_text().strip().split("\n")[1:]
    positions = [{"label": r.split(",", 1)[0].strip()} for r in rows]

    print(
        f"\n{'Label':<10} {'Dist(mm)':>9} "
        + "  ".join(f"{'m'+str(k)+' σ_ei(mm)':>12}" for k in range(n_markers))
        + f"  {'avg σ_ei(mm)':>13}"
    )
    print("─" * (10 + 10 + n_markers * 14 + 15))

    report_lines = []
    dist_list, avg_ei_list = [], []

    for pos in positions:
        pos_dir = DATA_DIR / pos["label"]
        if not (pos_dir / "bodyA.csv").exists():
            print(f"  {pos['label']}: data not found, skipping")
            continue

        samples_A = load_poses_csv(str(pos_dir / "bodyA.csv"))
        samples_B = load_poses_csv(str(pos_dir / "bodyB.csv"))

        mean_A, C_A = se3_empirical_stats(samples_A)
        sA = summary_stats(mean_A, C_A)
        dist_mm = sA["distance_mm"]

        ei = compute_ei(samples_A, samples_B, fid_drill_local)

        per_marker_sigma = [marker_sigma_mm(ei[:, k, :]) for k in range(n_markers)]
        avg_sigma = float(np.mean(per_marker_sigma))

        sigma_str = "  ".join(f"{s:>12.4f}" for s in per_marker_sigma)
        print(f"{pos['label']:<10} {dist_mm:>9.0f}  {sigma_str}  {avg_sigma:>13.4f}")

        report_lines.append(
            f"{pos['label']} (dist={dist_mm:.0f} mm): "
            + " ".join(f"m{k}={per_marker_sigma[k]:.4f}mm" for k in range(n_markers))
            + f" avg={avg_sigma:.4f}mm"
        )
        dist_list.append(dist_mm)
        avg_ei_list.append(avg_sigma)

    (RESULTS_DIR / "ei_report.txt").write_text("\n".join(report_lines))
    print(f"\nSaved → {RESULTS_DIR / 'ei_report.txt'}")

    # ── Load tracker-frame marker sigmas from markersA.csv for comparison ──
    tracker_frame_list = []
    for pos in positions:
        pos_dir = DATA_DIR / pos["label"]
        mA_path = pos_dir / "markersA.csv"
        if not mA_path.exists():
            tracker_frame_list.append(None)
            continue
        data = np.loadtxt(str(mA_path), delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        markers = data.reshape(-1, n_markers, 3)
        sigmas = [marker_sigma_mm(markers[:, k, :]) for k in range(n_markers)]
        tracker_frame_list.append(float(np.mean(sigmas)))

    if len(dist_list) >= 2:
        order = np.argsort(dist_list)
        dists   = np.array(dist_list)[order]
        ei_sigs = np.array(avg_ei_list)[order]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(dists, ei_sigs, "o-", color="steelblue",
                label="avg σ (anatomy frame, camera-motion corrected)")

        # Overlay tracker-frame sigma if available
        tf_vals = [tracker_frame_list[i] for i in order]
        if any(v is not None for v in tf_vals):
            tf_clean = [v if v is not None else np.nan for v in tf_vals]
            ax.plot(dists, tf_clean, "o--", color="tomato",
                    label="avg σ (tracker frame, raw)")

        ax.set_xlabel("Drill distance from tracker (mm, auto-computed)")
        ax.set_ylabel("Average per-marker σ (mm)")
        ax.set_title(
            "Drill marker random error: tracker frame vs anatomy frame\n"
            r"$e_i(t) = F_\mathrm{ref}(t)^{-1} \cdot b_i(t)$"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = RESULTS_DIR / "fig_ei_sigma_vs_distance.png"
        fig.savefig(str(out), dpi=150)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
