# Author: X.M. Christine Zhu

"""
Experiment 1 — Per-marker 3×3 covariance from raw tracker measurements.

Each body is held STILL at each position. The variation across N samples is
pure random noise (plus any residual camera motion):

    C_k = cov(p_k(1), ..., p_k(N))     shape (3, 3), metres²

Key questions:
  1. What is σ_ball per marker?  (scalar, mm)
  2. Is the noise isotropic?     (λ_min / λ_max close to 1)
  3. Does σ_ball change with drill distance from tracker?
  4. What is the representative C_ball to use in the OpticalTracker simulator?

Produces
--------
  results_fixed_drill/marker_covariance_report.txt
  results_fixed_drill/fig_sigma_ball_vs_distance.png
  results_fixed_drill/fig_marker_isotropy.png
  experiment/shared_cal/C_ball.json   ← average C_ball for OpticalTracker
"""

import sys
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve().parent
_EXP  = _HERE.parent
sys.path.insert(0, str(_EXP))

from utils.marker_io import load_marker_csv

DATA_DIR    = _HERE / "data_fixed_drill"
RESULTS_DIR = _HERE / "results_fixed_drill"
SHARED_CAL  = _EXP / "shared_cal"


def marker_covariance_stats(markers: np.ndarray) -> list:
    """
    Per-marker 3×3 covariance and eigendecomposition.

    Parameters
    ----------
    markers : ndarray, shape (N, n_markers, 3)   units: metres

    Returns
    -------
    list of dict (one per marker):
        mean        : (3,)   mean position in tracker frame, metres
        C           : (3,3)  empirical covariance, metres²
        sigma_mm    : float  sqrt(trace(C)/3) * 1000
        eigvals_mm2 : (3,)   eigenvalues in mm² (ascending)
        eigvecs     : (3,3)  corresponding eigenvectors (columns)
        isotropy    : float  λ_min / λ_max   (1.0 = perfectly isotropic)
    """
    out = []
    for k in range(markers.shape[1]):
        pts = markers[:, k, :]                      # (N, 3) metres
        mean = pts.mean(axis=0)
        C    = np.cov(pts, rowvar=False)            # (3, 3) metres²
        sigma_mm = float(np.sqrt(np.trace(C) / 3.0) * 1000.0)
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 0.0)
        isotropy = float(eigvals[0] / eigvals[2]) if eigvals[2] > 1e-30 else 0.0
        out.append({
            "mean":        mean,
            "C":           C,
            "sigma_mm":    sigma_mm,
            "eigvals_mm2": eigvals * 1e6,
            "eigvecs":     eigvecs,
            "isotropy":    isotropy,
        })
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SHARED_CAL.mkdir(parents=True, exist_ok=True)

    pos_dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())

    report_lines = ["position,body,marker,sigma_mm,lam1_mm2,lam2_mm2,lam3_mm2,isotropy"]
    dist_mm_list = []
    sigA_per_pos, sigB_per_pos     = [], []
    isoA_per_pos, isoB_per_pos     = [], []
    all_C_A,      all_C_B          = [], []

    hdr = (f"\n{'Position':<10} {'Body':<8} {'Marker':>6} "
           f"{'σ(mm)':>8} {'λ1(mm²)':>11} {'λ2(mm²)':>11} {'λ3(mm²)':>11} {'Isotropy':>10}")
    print(hdr)
    print("─" * 75)

    for pos_dir in pos_dirs:
        rawA = pos_dir / "markersA_raw.csv"
        rawB = pos_dir / "markersB_raw.csv"
        if not (rawA.exists() and rawB.exists()):
            print(f"  {pos_dir.name}: markersA/B_raw.csv missing — skipping")
            continue

        markers_A = load_marker_csv(rawA)   # (N, n_markers, 3) metres  — Drill
        markers_B = load_marker_csv(rawB)   # (N, n_markers, 3) metres  — Anatomy

        stats_A = marker_covariance_stats(markers_A)
        stats_B = marker_covariance_stats(markers_B)

        # Distance from tracker to drill body (auto-computed from data)
        centroid_A = np.mean([s["mean"] for s in stats_A], axis=0)
        dist_mm = float(np.linalg.norm(centroid_A)) * 1000.0
        dist_mm_list.append(dist_mm)

        sigA_per_pos.append([s["sigma_mm"]  for s in stats_A])
        sigB_per_pos.append([s["sigma_mm"]  for s in stats_B])
        isoA_per_pos.append([s["isotropy"]  for s in stats_A])
        isoB_per_pos.append([s["isotropy"]  for s in stats_B])
        all_C_A.extend([s["C"] for s in stats_A])
        all_C_B.extend([s["C"] for s in stats_B])

        for tag, stats in [("drill", stats_A), ("anatomy", stats_B)]:
            for k, s in enumerate(stats):
                lam = s["eigvals_mm2"]
                print(f"{pos_dir.name:<10} {tag:<8} {k:>6} "
                      f"{s['sigma_mm']:>8.4f} {lam[0]:>11.5f} {lam[1]:>11.5f} {lam[2]:>11.5f} "
                      f"{s['isotropy']:>10.4f}")
                report_lines.append(
                    f"{pos_dir.name},{tag},m{k},"
                    f"{s['sigma_mm']:.5f},{lam[0]:.6f},{lam[1]:.6f},{lam[2]:.6f},{s['isotropy']:.5f}"
                )

    (RESULTS_DIR / "marker_covariance_report.txt").write_text("\n".join(report_lines))
    print(f"\nSaved → {RESULTS_DIR / 'marker_covariance_report.txt'}")

    if not dist_mm_list:
        print("No raw marker data found. Run collect.py first.")
        return

    # ── Average C_ball ────────────────────────────────────────────────────────
    C_ball_drill   = np.mean(all_C_A, axis=0)   # averaged over all markers & positions
    C_ball_anatomy = np.mean(all_C_B, axis=0)
    C_ball_combined = 0.5 * (C_ball_drill + C_ball_anatomy)

    sigma_drill_mm   = float(np.sqrt(np.trace(C_ball_drill)   / 3.0) * 1000.0)
    sigma_anatomy_mm = float(np.sqrt(np.trace(C_ball_anatomy) / 3.0) * 1000.0)
    sigma_combined_mm = float(np.sqrt(np.trace(C_ball_combined) / 3.0) * 1000.0)

    print(f"\n── Average C_ball (drill,   {len(all_C_A)} markers total) ──")
    print(np.array2string(C_ball_drill * 1e6, precision=5, suppress_small=True))
    print(f"   σ_ball = {sigma_drill_mm:.4f} mm")

    print(f"\n── Average C_ball (anatomy, {len(all_C_B)} markers total) ──")
    print(np.array2string(C_ball_anatomy * 1e6, precision=5, suppress_small=True))
    print(f"   σ_ball = {sigma_anatomy_mm:.4f} mm")

    print(f"\n── Combined C_ball (average of both bodies) ──")
    print(np.array2string(C_ball_combined * 1e6, precision=5, suppress_small=True))
    print(f"   σ_ball = {sigma_combined_mm:.4f} mm")

    C_ball_out = {
        "C_ball_m2":         C_ball_combined.tolist(),
        "C_ball_mm2":        (C_ball_combined * 1e6).tolist(),
        "sigma_mm":          sigma_combined_mm,
        "C_ball_drill_m2":   C_ball_drill.tolist(),
        "C_ball_anatomy_m2": C_ball_anatomy.tolist(),
        "source":            "exp1_fixed_drill — avg over all markers and positions",
    }
    out_json = SHARED_CAL / "C_ball.json"
    out_json.write_text(json.dumps(C_ball_out, indent=2))
    print(f"\nSaved → {out_json}")

    # ── Plot: σ_ball vs drill distance ────────────────────────────────────────
    order     = np.argsort(dist_mm_list)
    dists     = np.array(dist_mm_list)[order]
    sigA_arr  = np.array(sigA_per_pos)[order]   # (n_pos, n_markers)
    sigB_arr  = np.array(sigB_per_pos)[order]
    n_markers = sigA_arr.shape[1]

    colors_A = ["tomato",    "salmon",       "indianred",   "lightcoral"]
    colors_B = ["steelblue", "cornflowerblue","royalblue",  "lightsteelblue"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for k in range(n_markers):
        ax.plot(dists, sigA_arr[:, k], "o-",  color=colors_A[k % 4], label=f"drill  m{k}")
        ax.plot(dists, sigB_arr[:, k], "s--", color=colors_B[k % 4], label=f"anatomy m{k}")
    ax.axhline(sigma_combined_mm, color="black", linestyle=":", linewidth=1.5,
               label=f"avg σ_ball = {sigma_combined_mm:.3f} mm")
    ax.set_xlabel("Drill distance from tracker (mm, auto-computed)")
    ax.set_ylabel("Per-marker σ (mm)")
    ax.set_title("Raw marker noise σ_ball vs. drill distance\n(each line = one fiducial)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out1 = RESULTS_DIR / "fig_sigma_ball_vs_distance.png"
    fig.savefig(str(out1), dpi=150)
    print(f"Saved → {out1}")

    # ── Plot: isotropy check ──────────────────────────────────────────────────
    isoA_arr = np.array(isoA_per_pos)[order]
    isoB_arr = np.array(isoB_per_pos)[order]

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    for k in range(n_markers):
        ax2.plot(dists, isoA_arr[:, k], "o-",  color=colors_A[k % 4], label=f"drill  m{k}")
        ax2.plot(dists, isoB_arr[:, k], "s--", color=colors_B[k % 4], label=f"anatomy m{k}")
    ax2.axhline(1.0, color="black", linestyle=":", linewidth=1, label="perfect isotropy")
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel("Drill distance from tracker (mm)")
    ax2.set_ylabel("Isotropy  λ_min / λ_max  (1 = isotropic)")
    ax2.set_title("Marker noise isotropy vs. drill distance")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = RESULTS_DIR / "fig_marker_isotropy.png"
    fig2.savefig(str(out2), dpi=150)
    print(f"Saved → {out2}")

    plt.show()


if __name__ == "__main__":
    main()
