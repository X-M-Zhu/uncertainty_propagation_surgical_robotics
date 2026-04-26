# Author: X.M. Christine Zhu
# Date: 02/06/2026

"""
scripts/validate_point_to_point_mc.py

Author: X.M. Christine Zhu
Date: Spring 2026

Monte Carlo validation for correlation-aware point-to-point covariance.

We build a small network:
    A --(uncertain)--> B

Two points p1, p2 attached to A. We query both into B and compute:
    delta = p2_B - p1_B

Analytic:
    - correlation-aware C_delta from net.query_relative_vector(...)
    - independent baseline from net.query_relative_vector_independent(...)

Monte Carlo:
    - sample eta ~ N(0, C_ab)
    - compute T_ab_sample = Exp(eta) @ T_ab_nom
    - transform points; build delta samples; compute sample covariance
"""

import numpy as np

from uncertainty_networks import UncertainTransform, GeometricNetwork
from uncertainty_networks.se3 import make_se3, rotz, exp_se3


def cov_sample(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


def main():
    seed = 3
    N = 50000
    rng = np.random.default_rng(seed)

    net = GeometricNetwork()

    T_ab_nom = make_se3(rotz(0.1), [0.2, 0.0, 0.0])
    C_ab = np.diag([5e-6] * 6)
    U_ab = UncertainTransform(T_ab_nom, C_ab)
    net.add_edge("A", "B", U_ab, add_inverse=True)

    # points in A with intrinsic covariance
    CpA = 1e-6 * np.eye(3)
    p1A = np.array([0.1, 0.0, 0.0])
    p2A = np.array([0.2, 0.0, 0.0])

    net.add_point("p1", "A", p1A, CpA)
    net.add_point("p2", "A", p2A, CpA)

    # analytic
    delta_ind, C_ind = net.query_relative_vector_independent("p1", "p2", "B")
    delta_corr, C_corr = net.query_relative_vector("p1", "p2", "B")

    # monte carlo
    deltas = np.zeros((N, 3), dtype=float)
    mean0 = np.zeros(6, dtype=float)
    mean0_3 = np.zeros(3, dtype=float)

    for i in range(N):
        eta = rng.multivariate_normal(mean0, C_ab)
        T_ab = exp_se3(eta) @ T_ab_nom
        R = T_ab[:3, :3]
        t = T_ab[:3, 3]

        dp1 = rng.multivariate_normal(mean0_3, CpA)
        dp2 = rng.multivariate_normal(mean0_3, CpA)

        p1B = R @ (p1A + dp1) + t
        p2B = R @ (p2A + dp2) + t
        deltas[i, :] = (p2B - p1B)

    C_mc = cov_sample(deltas)

    # comparisons
    def rel_frob(A, B):
        return np.linalg.norm(A - B, ord="fro") / np.linalg.norm(B, ord="fro")

    err_ind = rel_frob(C_ind, C_mc)
    err_corr = rel_frob(C_corr, C_mc)

    print("\n=== Point-to-point MC validation (correlation-aware) ===")
    print(f"Seed: {seed}, N={N}")
    print(f"||delta_ind - delta_corr|| = {np.linalg.norm(delta_ind - delta_corr):.3e}")
    print(f"rel frob error (independent) : {err_ind:.4f}")
    print(f"rel frob error (corr-aware)  : {err_corr:.4f}\n")

    np.set_printoptions(precision=6, suppress=False)
    print("diag(C_mc)   :", np.diag(C_mc))
    print("diag(C_ind)  :", np.diag(C_ind))
    print("diag(C_corr) :", np.diag(C_corr))


if __name__ == "__main__":
    main()
