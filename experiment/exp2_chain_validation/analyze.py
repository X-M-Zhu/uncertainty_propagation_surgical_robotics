# Author: X.M. Christine Zhu

"""
Experiment 2 — Analysis: Propagation chain validation.

For each configuration, compares:

  PREDICTED:  compose the two individual uncertain transforms using the
              framework  →  UncertainTransform.compose()

  EMPIRICAL:  compute the composed transform directly from the joint
              samples  (inv(T_A_sample) @ T_B_sample for each pair)

If the first-order propagation formula is correct, the predicted and
empirical covariances should be close.

Produces
--------
  results/cfg_XX_comparison.txt  — per-config numerical comparison
  results/fig_frobenius_error.png — Frobenius error of C_predicted vs C_empirical
"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve().parent
_EXP  = _HERE.parent
_ROOT = _EXP.parent
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(_ROOT / "src"))

from utils.se3_stats import load_poses_csv, se3_empirical_stats
from uncertainty_networks import UncertainTransform
from uncertainty_networks.se3 import inv_se3

DATA_DIR    = _HERE / "data"
RESULTS_DIR = _HERE / "results"


def relative_poses(samples_A: np.ndarray, samples_B: np.ndarray) -> np.ndarray:
    """
    Compute paired relative transforms:  T_AB[i] = inv(T_A[i]) @ T_B[i]

    If N_A != N_B, use min(N_A, N_B) pairs.
    """
    n = min(len(samples_A), len(samples_B))
    rel = np.array([inv_se3(samples_A[i]) @ samples_B[i] for i in range(n)])
    return rel


def analyse_config(cfg_dir: pathlib.Path) -> dict:
    samples_A = load_poses_csv(str(cfg_dir / "bodyA.csv"))
    samples_B = load_poses_csv(str(cfg_dir / "bodyB.csv"))
    n = min(len(samples_A), len(samples_B))
    samples_A, samples_B = samples_A[:n], samples_B[:n]

    # Empirical stats for each body in the tracker frame
    mean_A, C_A = se3_empirical_stats(samples_A)   # C of T_TA
    mean_B, C_B = se3_empirical_stats(samples_B)   # C of T_TB  ← empirical ground truth

    # Relative transform T_AB = inv(T_TA) @ T_TB, computed from simultaneous pairs.
    # Using paired samples is required: each row i of samples_A and samples_B was
    # captured at the same time step (see collect.py).
    rel_samples      = relative_poses(samples_A, samples_B)   # shape (n, 4, 4)
    mean_AB, C_AB    = se3_empirical_stats(rel_samples)       # C of T_AB

    # Framework prediction: compose(T_TA, T_AB) → predicted C of T_TB
    U_A    = UncertainTransform(mean_A,  C_A)
    U_AB   = UncertainTransform(mean_AB, C_AB)
    U_pred = U_A.compose(U_AB)

    # Compare predicted C_TB vs empirical C_TB (C_B)
    frob_error = np.linalg.norm(U_pred.C - C_B, "fro")
    rel_error  = frob_error / (np.linalg.norm(C_B, "fro") + 1e-30)

    return {
        "C_predicted": U_pred.C,
        "C_empirical": C_B,
        "mean_composed": mean_B,
        "frob_error":    frob_error,
        "rel_error_pct": rel_error * 100.0,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg_dirs = sorted(DATA_DIR.glob("cfg_*"))
    if not cfg_dirs:
        raise FileNotFoundError(f"No cfg_* folders found in {DATA_DIR}. Run collect.py first.")

    frob_errors = []
    print(f"{'Config':<12} {'Frob error':>12} {'Rel error (%)':>16}")
    print("─" * 44)

    for cfg_dir in cfg_dirs:
        label = cfg_dir.name
        try:
            res = analyse_config(cfg_dir)
        except FileNotFoundError as e:
            print(f"  {label}: skipping — {e}")
            continue

        print(f"{label:<12} {res['frob_error']:>12.6f} {res['rel_error_pct']:>15.2f}%")
        frob_errors.append(res["frob_error"])

        # Save per-config text report
        rpt = RESULTS_DIR / f"{label}_comparison.txt"
        with open(rpt, "w") as f:
            f.write(f"Configuration: {label}\n")
            f.write(f"Frobenius error: {res['frob_error']:.6f}\n")
            f.write(f"Relative error:  {res['rel_error_pct']:.2f}%\n\n")
            f.write("C_predicted (6×6):\n")
            f.write(np.array2string(res["C_predicted"], precision=6) + "\n\n")
            f.write("C_empirical (6×6):\n")
            f.write(np.array2string(res["C_empirical"], precision=6) + "\n")

    if frob_errors:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(1, len(frob_errors) + 1), frob_errors, color="steelblue", alpha=0.8)
        ax.set_xlabel("Configuration index")
        ax.set_ylabel("Frobenius norm  ||C_predicted − C_empirical||")
        ax.set_title("Predicted vs. empirical covariance: chain validation")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(RESULTS_DIR / "fig_frobenius_error.png"), dpi=150)
        print(f"\nSaved → {RESULTS_DIR / 'fig_frobenius_error.png'}")
        plt.show()


if __name__ == "__main__":
    main()
