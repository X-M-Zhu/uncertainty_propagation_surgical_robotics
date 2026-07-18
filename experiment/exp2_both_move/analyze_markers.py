# Author: X.M. Christine Zhu

"""
Experiment 2 — Marker analysis: per-marker empirical statistics.

Reads the raw per-marker positions collected by collect.py (straight from
the tracker's `marker_positions` topic — see utils/atracsys_interface.py)
and reports, for each tracked body's individual markers at each
configuration:
  - empirical mean position (tracker frame)
  - empirical 3x3 covariance
  - sigma (sqrt(trace(C)/3)) in mm

This is a sanity-check / visualization layer on top of the frame-level
analysis in analyze.py — it shows whether marker-level spread is consistent
across the four fiducials on each tool.

Produces
--------
  results/marker_report.txt
"""

import sys
import pathlib
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_EXP  = _HERE.parent
sys.path.insert(0, str(_EXP))

from utils.marker_io import load_marker_csv

DATA_DIR    = _HERE / "data"
RESULTS_DIR = _HERE / "results"


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
        raise FileNotFoundError(f"Run collect.py first. Expected: {DATA_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg_dirs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    report_lines = []

    print(f"\n{'Config':<10} {'Body':<8} {'Marker':>6} {'σ(mm)':>8}")
    print("─" * 40)

    for cfg_dir in cfg_dirs:
        markersA_path = cfg_dir / "markersA_raw.csv"
        markersB_path = cfg_dir / "markersB_raw.csv"
        if not (markersA_path.exists() and markersB_path.exists()):
            print(f"  {cfg_dir.name}: no markersA_raw.csv/markersB_raw.csv, skipping "
                  f"(re-run collect.py to get raw per-marker data)")
            continue

        markers_A = load_marker_csv(markersA_path)
        markers_B = load_marker_csv(markersB_path)
        stats_A = marker_stats(markers_A)
        stats_B = marker_stats(markers_B)

        for k, s in enumerate(stats_A):
            print(f"{cfg_dir.name:<10} {'Anatomy':<8} {k:>6} {s['sigma_mm']:>8.4f}")
            report_lines.append(f"{cfg_dir.name},Anatomy,m{k},{s['sigma_mm']:.4f}")
        for k, s in enumerate(stats_B):
            print(f"{cfg_dir.name:<10} {'Drill':<8} {k:>6} {s['sigma_mm']:>8.4f}")
            report_lines.append(f"{cfg_dir.name},Drill,m{k},{s['sigma_mm']:.4f}")

    (RESULTS_DIR / "marker_report.txt").write_text(
        "\n".join(["config,body,marker,sigma_mm"] + report_lines)
    )
    print(f"\nSaved → {RESULTS_DIR / 'marker_report.txt'}")


if __name__ == "__main__":
    main()
