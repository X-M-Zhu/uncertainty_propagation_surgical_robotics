# Author: X.M. Christine Zhu

"""
Experiment 1 — Pivot calibration solve: derive the drill's local tip offset.

Standard two-unknown pivot calibration. While the tip is held fixed in world
space and the drill rotates around it, every sample obeys:

    R_i @ p_tip_local + t_i = p_pivot_world

for the same (unknown) p_tip_local (fixed in the drill's own frame) and
p_pivot_world (fixed in the tracker frame). Stacking over all N samples:

    [R_i | -I_3] [p_tip_local; p_pivot_world] = -t_i

gives a 3N x 6 linear system, solved by least squares for both unknowns at
once.

Reads
-----
  data_fixed_drill/pivot_cal/bodyA_pivot.csv   (from collect_pivot.py)

Writes
------
  data_fixed_drill/pivot_cal/tip_offset.json
      {"x_mm":, "y_mm":, "z_mm":, "rms_residual_mm":, "n_samples":}

Note: this intentionally does NOT modify geometry_anspoch_drill.json's
pivot field — that file is live SDK configuration shared on the lab
machine, not something this analysis should mutate.
"""

import sys
import json
import pathlib
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_EXP = _HERE.parent
sys.path.insert(0, str(_EXP))

from utils.se3_stats import load_poses_csv

PIVOT_DIR = _HERE / "data_fixed_drill" / "pivot_cal"


def solve_pivot(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Parameters
    ----------
    samples : ndarray, shape (N, 4, 4)

    Returns
    -------
    p_tip_local : ndarray, shape (3,)   — tip offset in the drill's own frame
    p_pivot_world : ndarray, shape (3,) — fixed pivot point in tracker frame
    rms_residual : float                — RMS of the linear-solve residual (m)
    """
    N = samples.shape[0]
    A = np.zeros((3 * N, 6), dtype=np.float64)
    b = np.zeros(3 * N, dtype=np.float64)
    for i in range(N):
        R = samples[i, :3, :3]
        t = samples[i, :3, 3]
        A[3 * i : 3 * i + 3, :3] = R
        A[3 * i : 3 * i + 3, 3:] = -np.eye(3)
        b[3 * i : 3 * i + 3] = -t

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    p_tip_local = x[:3]
    p_pivot_world = x[3:]

    residuals = A @ x - b
    rms_residual = float(np.sqrt(np.mean(residuals**2)))
    return p_tip_local, p_pivot_world, rms_residual


def main():
    csv_path = PIVOT_DIR / "bodyA_pivot.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run collect_pivot.py first. Expected: {csv_path}")

    samples = load_poses_csv(str(csv_path))
    p_tip_local, p_pivot_world, rms_residual = solve_pivot(samples)

    print(f"Solved from {len(samples)} samples.")
    print(f"  Tip offset (drill-local, mm): {p_tip_local * 1000.0}")
    print(f"  Pivot point (tracker frame, mm): {p_pivot_world * 1000.0}")
    print(f"  RMS residual: {rms_residual * 1000.0:.4f} mm")

    out = {
        "x_mm": float(p_tip_local[0] * 1000.0),
        "y_mm": float(p_tip_local[1] * 1000.0),
        "z_mm": float(p_tip_local[2] * 1000.0),
        "rms_residual_mm": rms_residual * 1000.0,
        "n_samples": int(len(samples)),
    }
    out_path = PIVOT_DIR / "tip_offset.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {out_path}")

    if rms_residual * 1000.0 > 1.0:
        print(
            "\nWARNING: RMS residual > 1 mm — calibration quality is poor. "
            "Consider re-collecting with more angular spread (collect_pivot.py)."
        )


if __name__ == "__main__":
    main()
