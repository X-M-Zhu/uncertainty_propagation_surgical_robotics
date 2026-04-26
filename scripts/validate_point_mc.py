# Author: X.M. Christine Zhu
# Date: 02/06/2026

"""
Monte Carlo validation for point uncertainty propagation under CIS I convention.

Goal:
Validate the point covariance propagation formula used in UncertainTransform.transform_point:

Nominal transform:
    p'_nom = R p + t

CIS I left-perturbation model:
    T_true = Exp(eta) ∘ T_nom,   eta = [alpha; epsilon] ~ N(0, C)

First-order linearization (CIS I):
    δp' ≈ -[p'_nom]× alpha + epsilon

Thus Jacobians:
    J_eta = [ -[p'_nom]×   I ]    (3×6)
    J_p   = R                     (3×3)

Covariance propagation:
    Cp' ≈ J_eta C J_eta^T + R Cp R^T
(If Cp is None, compare pose-only term: Cp' ≈ J_eta C J_eta^T)

Monte Carlo
Simulate N samples:
    T_i = Exp(eta_i) ∘ T_nom
    p_i = (T_i) * (p + δp_i)

Compute sample covariance of {p_i} and compare with analytical Cp'.
"""

import numpy as np

from uncertainty_networks.se3 import make_se3, rotz, exp_se3, skew
from uncertainty_networks import UncertainTransform


def cov_sample(X: np.ndarray) -> np.ndarray:
    """Sample covariance for row-stacked data X of shape (N, d)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


def main():
    # Config
    seed = 11
    N = 50000
    do_plot = True

    rng = np.random.default_rng(seed)

    # Nominal transform T_nom = (R, t)
    T_nom = make_se3(rotz(np.deg2rad(15.0)), [0.10, -0.03, 0.02])
    R = T_nom[:3, :3]
    t = T_nom[:3, 3]

    # Pose covariance C (eta = [alpha; epsilon])
    C = np.diag([2e-6, 2e-6, 2e-6,  5e-6, 5e-6, 5e-6])

    # Point + intrinsic covariance
    p = np.array([0.08, 0.02, -0.01])

    # Set Cp to None if want pose-only validation
    Cp = 3e-6 * np.eye(3)

    U = UncertainTransform(T_nom, C)

    # Analytical covariance from implementation
    p_nom, Cp_an = U.transform_point(p, Cp=Cp)

    # Monte Carlo simulation
    mean0_6 = np.zeros(6, dtype=float)
    mean0_3 = np.zeros(3, dtype=float)

    p_samples = np.zeros((N, 3), dtype=float)

    for i in range(N):
        eta = rng.multivariate_normal(mean=mean0_6, cov=C)
        T_i = exp_se3(eta) @ T_nom

        # intrinsic point noise
        if Cp is None:
            p_i_local = p
        else:
            dp = rng.multivariate_normal(mean=mean0_3, cov=Cp)
            p_i_local = p + dp

        # apply transform
        R_i = T_i[:3, :3]
        t_i = T_i[:3, 3]
        p_samples[i, :] = R_i @ p_i_local + t_i

    Cp_mc = cov_sample(p_samples)

    # Compare
    rel_frob = np.linalg.norm(Cp_mc - Cp_an, ord="fro") / np.linalg.norm(Cp_an, ord="fro")

    print("\n=== Point Monte Carlo Validation ===")
    print(f"Seed: {seed}")
    print(f"N samples: {N}")
    print(f"Relative Frobenius error: {rel_frob:.4f}\n")

    np.set_printoptions(precision=9, suppress=False)
    print("p_nom:", p_nom)
    print("diag(Cp_analytic):", np.diag(Cp_an))
    print("diag(Cp_mc)      :", np.diag(Cp_mc))

    # Plot: diagonal comparison
    if do_plot:
        import matplotlib.pyplot as plt

        idx = np.arange(3)
        plt.figure()
        plt.plot(idx, np.diag(Cp_an), marker="o", label="analytic")
        plt.plot(idx, np.diag(Cp_mc), marker="x", label="monte carlo")
        plt.xlabel("component index (0..2) for point (x,y,z)")
        plt.ylabel("variance")
        plt.title("Point covariance diagonal: analytic vs Monte Carlo")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
