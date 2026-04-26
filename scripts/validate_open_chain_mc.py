# Author: X.M. Christine Zhu
# Date: 02/06/2026

"""
Monte Carlo validation for SE(3) uncertainty propagation on an open chain.]
This scipt generates a synthetic open chain consisting of two uncertain SE(3) transforms with specified covariances. It then compares analytic covariance propagation with Monte Carlo sampling of the perturbation model.

Goal:
Verify the first-order analytical covariance propagation rule from the math note:

Given two uncertain transforms (independent perturbations):
    F_ab = {F_nom,ab, C_ab}
    F_bc = {F_nom,bc, C_bc}

Nominal composition:
    F_nom,ac = F_nom,ab ∘ F_nom,bc

Analytical covariance propagation (CIS I, left perturbation):
    C_ac ≈ C_ab + Ad_{F_nom,ab} C_bc Ad_{F_nom,ab}^T

Monte Carlo "ground truth"
We simulate samples using the left-multiplicative perturbation model:
    T_ab = Exp(eta_ab) ∘ F_nom,ab
    T_bc = Exp(eta_bc) ∘ F_nom,bc
    T_ac = T_ab ∘ T_bc

Then compute residuals in tangent space:
    xi_i = Log( T_ac^{(i)} ∘ (F_nom,ac)^{-1} )

The sample covariance of {xi_i} is compared to the analytical C_ac.

Outputs:
- Prints analytic covariance, MC covariance, and relative Frobenius error.
- Produces a simple plot comparing diagonal terms.
"""

import numpy as np

from uncertainty_networks.se3 import make_se3, rotz, inv_se3, adjoint_se3, exp_se3, log_se3
from uncertainty_networks import UncertainTransform


def sample_gaussian(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample from N(mean, cov)."""
    return rng.multivariate_normal(mean=mean, cov=cov)


def cov_sample(X: np.ndarray) -> np.ndarray:
    """
    Sample covariance of row-stacked data X of shape (N, d).
    Returns (d, d).
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


def main():
    # Configuration
    seed = 7
    N = 20000  # number of Monte Carlo samples
    do_plot = True  # set False if don't want plots

    rng = np.random.default_rng(seed)

    # Define nominal transforms
    # F_ab: small rotation + translation
    F_ab_nom = make_se3(rotz(np.deg2rad(12.0)), [0.10, 0.02, -0.01])

    # F_bc: pure translation
    F_bc_nom = make_se3(np.eye(3), [0.00, 0.15, 0.03])

    # Define covariances (small, realistic scale for first-order validity)
    # Units:
    #   alpha in rad, epsilon in meters (consistent with CIS I ordering)
    C_ab = np.diag([2e-6, 2e-6, 2e-6,  5e-6, 5e-6, 5e-6])
    C_bc = np.diag([3e-6, 3e-6, 3e-6,  4e-6, 4e-6, 4e-6])

    U_ab = UncertainTransform(F_ab_nom, C_ab)
    U_bc = UncertainTransform(F_bc_nom, C_bc)

    # Analytical propagation
    U_ac_analytic = U_ab @ U_bc
    F_ac_nom = U_ac_analytic.F_nom
    C_ac_analytic = U_ac_analytic.C

    # Monte Carlo simulation
    F_ac_nom_inv = inv_se3(F_ac_nom)

    xi_samples = np.zeros((N, 6), dtype=float)

    mean0 = np.zeros(6, dtype=float)

    for i in range(N):
        eta_ab = sample_gaussian(mean0, C_ab, rng)
        eta_bc = sample_gaussian(mean0, C_bc, rng)

        T_ab = exp_se3(eta_ab) @ F_ab_nom
        T_bc = exp_se3(eta_bc) @ F_bc_nom

        T_ac = T_ab @ T_bc

        # Residual in tangent space: xi = Log( T_ac ∘ F_nom^{-1} )
        T_res = T_ac @ F_ac_nom_inv
        xi = log_se3(T_res)

        xi_samples[i, :] = xi

    C_ac_mc = cov_sample(xi_samples)

    # Comparison metrics
    frob_rel = np.linalg.norm(C_ac_mc - C_ac_analytic, ord="fro") / np.linalg.norm(C_ac_analytic, ord="fro")

    print("\n=== Open-chain Monte Carlo Validation ===")
    print(f"Seed: {seed}")
    print(f"N samples: {N}")
    print(f"Relative Frobenius error: {frob_rel:.4f}\n")

    np.set_printoptions(precision=9, suppress=True)
    print("diag(C_ac_analytic):", np.diag(C_ac_analytic))
    print("diag(C_ac_mc)      :", np.diag(C_ac_mc))

    # Plot
    if do_plot:
        import matplotlib.pyplot as plt

        idx = np.arange(6)
        plt.figure()
        plt.plot(idx, np.diag(C_ac_analytic), marker="o", label="analytic")
        plt.plot(idx, np.diag(C_ac_mc), marker="x", label="monte carlo")
        plt.xlabel("component index (0..5) for [alpha; epsilon]")
        plt.ylabel("variance")
        plt.title("Open-chain covariance diagonal: analytic vs Monte Carlo")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
