# Author: X.M. Christine Zhu

"""
Experiment 1 — Marker analysis: per-marker empirical statistics.

Reads the per-marker positions derived by extract_markers.py and reports,
for each tracked body's individual markers at each position:
  - empirical mean position (tracker frame)
  - empirical 3x3 covariance
  - sigma (sqrt(trace(C)/3)) in mm

This is a sanity-check / visualization layer on top of the frame-level
analysis in analyze.py — it shows whether marker-level spread is consistent
across the four fiducials on each tool (as the rigid-body assumption would
predict). It is NOT an independent measurement of true marker-level noise
(see extract_markers.py docstring for that caveat).

Produces
--------
  results_fixed_drill/marker_report.txt
  results_fixed_drill/fig_marker_sigma_vs_distance.png
"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve().parent
_EXP  = _HERE.parent
sys.path.insert(0, str(_EXP))

DATA_DIR    = _HERE / "data_fixed_drill"
RESULTS_DIR = _HERE / "results_fixed_drill"


def load_marker_csv(path: pathlib.Path) -> np.ndarray:
    """Load a flattened marker CSV back into (N, n_markers, 3)."""
    data = np.loadtxt(str(path), delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    n_markers = data.shape[1] // 3
    return data.reshape(-1, n_markers, 3)


def marker_stats(markers: np.ndarray) -> list:
    """
    Per-marker empirical mean/covariance/sigma.

    Parameters
    ----------
    markers : ndarray, shape (N, n_markers, 3)

    Returns
    -------
    list of dict, one per marker:
        {"mean": (3,), "C": (3,3), "sigma_mm": float}
    """
    n_markers = markers.shape[1]
    out = []
    for k in range(n_markers):
        pts = markers[:, k, :]
        mean = pts.mean(axis=0)
        C = np.cov(pts, rowvar=False)
        sigma_mm = float(np.sqrt(np.trace(C) / 3.0) * 1000.0)
        out.append({"mean": mean, "C": C, "sigma_mm": sigma_mm})
    return out


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Run collect.py and extract_markers.py first. Expected: {DATA_DIR}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pos_dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    report_lines = []
    dist_mm_list, sigA_list, sigB_list = [], [], []

    print(f"\n{'Position':<10} {'Body':<8} {'Marker':>6} {'σ(mm)':>8}")
    print("─" * 40)

    for pos_dir in pos_dirs:
        markersA_path = pos_dir / "markersA.csv"
        markersB_path = pos_dir / "markersB.csv"
        if not (markersA_path.exists() and markersB_path.exists()):
            print(f"  {pos_dir.name}: markers not extracted, skipping "
                  f"(run extract_markers.py)")
            continue

        markers_A = load_marker_csv(markersA_path)
        markers_B = load_marker_csv(markersB_path)
        stats_A = marker_stats(markers_A)
        stats_B = marker_stats(markers_B)

        for k, s in enumerate(stats_A):
            print(f"{pos_dir.name:<10} {'drill':<8} {k:>6} {s['sigma_mm']:>8.4f}")
            report_lines.append(f"{pos_dir.name},drill,m{k},{s['sigma_mm']:.4f}")
        for k, s in enumerate(stats_B):
            print(f"{pos_dir.name:<10} {'anatomy':<8} {k:>6} {s['sigma_mm']:>8.4f}")
            report_lines.append(f"{pos_dir.name},anatomy,m{k},{s['sigma_mm']:.4f}")

        # Distance auto-computed from the drill markers' mean centroid (no manual input).
        centroid_A = np.mean([s["mean"] for s in stats_A], axis=0)
        dist_mm  = float(np.linalg.norm(centroid_A)) * 1000.0
        avg_sigA = float(np.mean([s["sigma_mm"] for s in stats_A]))
        avg_sigB = float(np.mean([s["sigma_mm"] for s in stats_B]))
        dist_mm_list.append(dist_mm)
        sigA_list.append(avg_sigA)
        sigB_list.append(avg_sigB)

    (RESULTS_DIR / "marker_report.txt").write_text(
        "\n".join(["position,body,marker,sigma_mm"] + report_lines)
    )
    print(f"\nSaved → {RESULTS_DIR / 'marker_report.txt'}")

    if len(dist_mm_list) >= 2:
        order = np.argsort(dist_mm_list)
        dists = np.array(dist_mm_list)[order]
        sigA  = np.array(sigA_list)[order]
        sigB  = np.array(sigB_list)[order]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(dists, sigA, "o-", color="tomato",    label="avg marker σ — drill (moved)")
        ax.plot(dists, sigB, "o-", color="steelblue", label="avg marker σ — Anatomy (fixed)")
        ax.set_xlabel("Drill distance from tracker (mm, auto-computed)")
        ax.set_ylabel("Average per-marker σ (mm)")
        ax.set_title("Per-marker positional spread vs drill distance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = RESULTS_DIR / "fig_marker_sigma_vs_distance.png"
        fig.savefig(str(out), dpi=150)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
