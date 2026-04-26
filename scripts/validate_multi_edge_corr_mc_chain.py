# Author: X.M. Christine Zhu
# Date: 02/06/2026

import numpy as np

from uncertainty_networks import UncertainTransform, GeometricNetwork
from uncertainty_networks.se3 import make_se3, rotz, exp_se3


def cov_sample(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


def rel_frob(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(A - B, ord="fro") / np.linalg.norm(B, ord="fro"))


def main():
    seed = 5
    N = 60000
    rng = np.random.default_rng(seed)

    net = GeometricNetwork()

    # Chain: A -> B -> C -> D
    # Keep covariances small (first-order valid)
    C1 = np.diag([3e-6] * 6)
    C2 = np.diag([4e-6] * 6)
    C3 = np.diag([5e-6] * 6)

    T_ab = make_se3(rotz(0.07), [0.10, 0.00, 0.00])
    T_bc = make_se3(rotz(-0.04), [0.00, 0.12, 0.00])
    T_cd = make_se3(rotz(0.03), [0.00, 0.00, 0.08])

    U_ab = UncertainTransform(T_ab, C1)
    U_bc = UncertainTransform(T_bc, C2)
    U_cd = UncertainTransform(T_cd, C3)

    net.add_edge("A", "B", U_ab, add_inverse=True)
    net.add_edge("B", "C", U_bc, add_inverse=True)
    net.add_edge("C", "D", U_cd, add_inverse=True)

    # Points: p1 in A, p2 in C, query in D
    Cp = 2e-6 * np.eye(3)
    p1A = np.array([0.05, 0.01, 0.00])
    p2C = np.array([0.02, -0.01, 0.01])

    net.add_point("p1", "A", p1A, Cp)
    net.add_point("p2", "C", p2C, Cp)

    # Analytic
    delta_ind, C_ind = net.query_relative_vector_independent("p1", "p2", "D")
    delta_corr, C_corr = net.query_relative_vector("p1", "p2", "D")

    # Monte Carlo
    deltas = np.zeros((N, 3), dtype=float)
    mean0_6 = np.zeros(6, dtype=float)
    mean0_3 = np.zeros(3, dtype=float)

    for i in range(N):
        eta1 = rng.multivariate_normal(mean0_6, C1)
        eta2 = rng.multivariate_normal(mean0_6, C2)
        eta3 = rng.multivariate_normal(mean0_6, C3)

        # Sampled transforms
        T_ab_s = exp_se3(eta1) @ T_ab
        T_bc_s = exp_se3(eta2) @ T_bc
        T_cd_s = exp_se3(eta3) @ T_cd

        T_ad_s = T_ab_s @ T_bc_s @ T_cd_s
        T_cd_only = T_cd_s

        # intrinsic point noise
        dp1 = rng.multivariate_normal(mean0_3, Cp)
        dp2 = rng.multivariate_normal(mean0_3, Cp)

        # p1: A -> D
        R_ad = T_ad_s[:3, :3]
        t_ad = T_ad_s[:3, 3]
        p1D = R_ad @ (p1A + dp1) + t_ad

        # p2: C -> D
        R_cd = T_cd_only[:3, :3]
        t_cd = T_cd_only[:3, 3]
        p2D = R_cd @ (p2C + dp2) + t_cd

        deltas[i, :] = (p2D - p1D)

    C_mc = cov_sample(deltas)

    err_ind = rel_frob(C_ind, C_mc)
    err_corr = rel_frob(C_corr, C_mc)

    print("\n=== Multi-edge correlation MC (CHAIN) ===")
    print(f"Seed={seed}, N={N}")
    print(f"||delta_ind - delta_corr|| = {np.linalg.norm(delta_ind - delta_corr):.3e}")
    print(f"rel frob error (independent) : {err_ind:.4f}")
    print(f"rel frob error (corr-aware)  : {err_corr:.4f}")
    print(f"trace(C_ind)  = {np.trace(C_ind):.6e}")
    print(f"trace(C_corr) = {np.trace(C_corr):.6e}")
    print(f"trace(C_mc)   = {np.trace(C_mc):.6e}")

    np.set_printoptions(precision=6, suppress=False)
    print("diag(C_mc)   :", np.diag(C_mc))
    print("diag(C_ind)  :", np.diag(C_ind))
    print("diag(C_corr) :", np.diag(C_corr))


if __name__ == "__main__":
    main()
