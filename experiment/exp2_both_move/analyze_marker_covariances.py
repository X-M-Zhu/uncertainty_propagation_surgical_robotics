# Author: X.M. Christine Zhu

"""
Experiment 2 — Per-marker 3×3 covariance from raw tracker measurements.

Within each configuration, both bodies are held STILL. The variation across
N samples is pure random noise:

    C_k = cov(p_k(1), ..., p_k(N))     shape (3, 3), metres²

Key questions:
  1. What is σ_ball per marker at each configuration?  (scalar, mm)
  2. Is the noise isotropic?     (λ_min / λ_max close to 1)
  3. Is C_ball consistent across all configurations?
  4. How does it compare to the C_ball estimated in exp1?

Produces
--------
  results/marker_covariance_report.txt
  results/fig_sigma_ball_by_config.png
  results/fig_marker_isotropy.png
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

DATA_DIR    = _HERE / "data"
RESULTS_DIR = _HERE / "results"
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
        pts = markers[:, k, :]
        mean = pts.mean(axis=0)
        C    = np.cov(pts, rowvar=False)
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

    cfg_dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())

    report_lines = ["config,body,marker,sigma_mm,lam1_mm2,lam2_mm2,lam3_mm2,isotropy"]
    labels = []
    sigA_per_cfg, sigB_per_cfg = [], []
    isoA_per_cfg, isoB_per_cfg = [], []
    all_C_A, all_C_B           = [], []

    hdr = (f"\n{'Config':<10} {'Body':<8} {'Marker':>6} "
           f"{'σ(mm)':>8} {'λ1(mm²)':>11} {'λ2(mm²)':>11} {'λ3(mm²)':>11} {'Isotropy':>10}")
    print(hdr)
    print("─" * 75)

    for cfg_dir in cfg_dirs:
        rawA = cfg_dir / "markersA_raw.csv"
        rawB = cfg_dir / "markersB_raw.csv"
        if not (rawA.exists() and rawB.exists()):
            print(f"  {cfg_dir.name}: markersA/B_raw.csv missing — skipping")
            continue

        markers_A = load_marker_csv(rawA)   # (N, n_markers, 3) metres  — Anatomy
        markers_B = load_marker_csv(rawB)   # (N, n_markers, 3) metres  — Drill

        stats_A = marker_covariance_stats(markers_A)
        stats_B = marker_covariance_stats(markers_B)

        labels.append(cfg_dir.name)
        sigA_per_cfg.append([s["sigma_mm"] for s in stats_A])
        sigB_per_cfg.append([s["sigma_mm"] for s in stats_B])
        isoA_per_cfg.append([s["isotropy"] for s in stats_A])
        isoB_per_cfg.append([s["isotropy"] for s in stats_B])
        all_C_A.extend([s["C"] for s in stats_A])
        all_C_B.extend([s["C"] for s in stats_B])

        for tag, stats in [("anatomy", stats_A), ("drill", stats_B)]:
            for k, s in enumerate(stats):
                lam = s["eigvals_mm2"]
                print(f"{cfg_dir.name:<10} {tag:<8} {k:>6} "
                      f"{s['sigma_mm']:>8.4f} {lam[0]:>11.5f} {lam[1]:>11.5f} {lam[2]:>11.5f} "
                      f"{s['isotropy']:>10.4f}")
                report_lines.append(
                    f"{cfg_dir.name},{tag},m{k},"
                    f"{s['sigma_mm']:.5f},{lam[0]:.6f},{lam[1]:.6f},{lam[2]:.6f},{s['isotropy']:.5f}"
                )

    (RESULTS_DIR / "marker_covariance_report.txt").write_text("\n".join(report_lines))
    print(f"\nSaved → {RESULTS_DIR / 'marker_covariance_report.txt'}")

    if not labels:
        print("No raw marker data found. Run collect.py first.")
        return

    # ── Average C_ball ────────────────────────────────────────────────────────
    C_ball_anatomy  = np.mean(all_C_A, axis=0)
    C_ball_drill    = np.mean(all_C_B, axis=0)
    C_ball_combined = 0.5 * (C_ball_anatomy + C_ball_drill)

    sigma_anatomy_mm  = float(np.sqrt(np.trace(C_ball_anatomy)  / 3.0) * 1000.0)
    sigma_drill_mm    = float(np.sqrt(np.trace(C_ball_drill)    / 3.0) * 1000.0)
    sigma_combined_mm = float(np.sqrt(np.trace(C_ball_combined) / 3.0) * 1000.0)

    print(f"\n── Average C_ball (anatomy, {len(all_C_A)} markers total) — (mm²) ──")
    print(np.array2string(C_ball_anatomy * 1e6, precision=5, suppress_small=True))
    print(f"   σ_ball = {sigma_anatomy_mm:.4f} mm")

    print(f"\n── Average C_ball (drill,   {len(all_C_B)} markers total) — (mm²) ──")
    print(np.array2string(C_ball_drill * 1e6, precision=5, suppress_small=True))
    print(f"   σ_ball = {sigma_drill_mm:.4f} mm")

    print(f"\n── Combined C_ball (average of both bodies) — (mm²) ──")
    print(np.array2string(C_ball_combined * 1e6, precision=5, suppress_small=True))
    print(f"   σ_ball = {sigma_combined_mm:.4f} mm")

    # Load exp1 C_ball for cross-experiment comparison
    exp1_json = SHARED_CAL / "C_ball.json"
    if exp1_json.exists():
        exp1_data = json.loads(exp1_json.read_text())
        print(f"\n── Comparison with exp1 C_ball ──")
        print(f"   exp1 σ_ball = {exp1_data['sigma_mm']:.4f} mm")
        print(f"   exp2 σ_ball = {sigma_combined_mm:.4f} mm")

    # ── Plot: σ_ball by configuration ────────────────────────────────────────
    x         = np.arange(len(labels))
    sigA_arr  = np.array(sigA_per_cfg)   # (n_cfg, n_markers)
    sigB_arr  = np.array(sigB_per_cfg)
    n_markers = sigA_arr.shape[1]

    colors_A = ["steelblue",   "cornflowerblue", "royalblue",  "lightsteelblue"]
    colors_B = ["tomato",      "salmon",         "indianred",  "lightcoral"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for k in range(n_markers):
        ax.plot(x, sigA_arr[:, k], "o-",  color=colors_A[k % 4], label=f"anatomy m{k}")
        ax.plot(x, sigB_arr[:, k], "s--", color=colors_B[k % 4], label=f"drill   m{k}")
    ax.axhline(sigma_combined_mm, color="black", linestyle=":", linewidth=1.5,
               label=f"avg σ_ball = {sigma_combined_mm:.3f} mm")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Per-marker σ (mm)")
    ax.set_title("Raw marker noise σ_ball by configuration\n(each line = one fiducial)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out1 = RESULTS_DIR / "fig_sigma_ball_by_config.png"
    fig.savefig(str(out1), dpi=150)
    print(f"Saved → {out1}")

    # ── Plot: isotropy check (bar chart) ─────────────────────────────────────
    isoA_arr = np.array(isoA_per_cfg)   # (n_cfg, n_markers)
    isoB_arr = np.array(isoB_per_cfg)

    avg_isoA = isoA_arr.mean(axis=1)
    avg_isoB = isoB_arr.mean(axis=1)

    x_bar = np.arange(len(labels))
    width = 0.35

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.bar(x_bar - width / 2, avg_isoA, width, color="steelblue", alpha=0.85, label="anatomy (avg across markers)")
    ax2.bar(x_bar + width / 2, avg_isoB, width, color="tomato",    alpha=0.85, label="drill   (avg across markers)")

    # Overlay individual marker values as dots
    for k in range(n_markers):
        ax2.scatter(x_bar - width / 2, isoA_arr[:, k], color="navy",    s=18, zorder=3)
        ax2.scatter(x_bar + width / 2, isoB_arr[:, k], color="darkred", s=18, zorder=3)

    ax2.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="perfect isotropy (= 1)")
    ax2.set_xticks(x_bar)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel("Isotropy  λ_min / λ_max")
    ax2.set_title("Marker noise isotropy by configuration\n(bar = avg across 4 markers, dots = individual markers)")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    out2 = RESULTS_DIR / "fig_marker_isotropy.png"
    fig2.savefig(str(out2), dpi=150)
    print(f"Saved → {out2}")

    plt.show()


if __name__ == "__main__":
    main()
