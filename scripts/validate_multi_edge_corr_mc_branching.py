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
    seed = 6
    N = 60000
    rng = np.random.default_rng(seed)

    net = GeometricNetwork()

    # Branching:
    # A -> B -> D
    # A -> C -> D
    C_ab = np.diag([3e-6] * 6)
    C_bd = np.diag([6e-6] * 6)
    C_ac = np.diag([3e-6] * 6)
    C_cd = np.diag([6e-6] * 6)

    T_ab = make_se3(rotz(0.05), [0.10, 0.00, 0.00])
    T_bd = make_se3(rotz(0.02), [0.00, 0.10, 0.02])
    T_ac = make_se3(rotz(-0.04), [0.10, 0.02, 0.00])
    T_cd = make_se3(rotz(0.01), [0.00, 0.08, 0.02])

    U_ab = UncertainTransform(T_ab, C_ab)
    U_bd = UncertainTransform(T_bd, C_bd)
    U_ac = UncertainTransform(T_ac, C_ac)
    U_cd = UncertainTransform(T_cd, C_cd)

    net.add_edge("A", "B", U_ab, add_inverse=True)
    net.add_edge("B", "D", U_bd, add_inverse=True)
    net.add_edge("A", "C", U_ac, add_inverse=True)
    net.add_edge("C", "D", U_cd, add_inverse=True)

    # Points: p1 on B, p2 on C, query in D
    Cp = 2e-6 * np.eye(3)
    p1B = np.array([0.02, 0.00, 0.00])
    p2C = np.array([-0.01, 0.01, 0.00])

    net.add_point("p1", "B", p1B, Cp)
    net.add_point("p2", "C", p2C, Cp)

    # Analytic
    delta_ind, C_ind = net.query_relative_vector_independent("p1", "p2", "D")
    delta_corr, C_corr = net.query_relative_vector("p1", "p2", "D")

    # Monte Carlo
    deltas = np.zeros((N, 3), dtype=float)
    mean0_6 = np.zeros(6, dtype=float)
    mean0_3 = np.zeros(3, dtype=float)

    for i in range(N):
        eta_ab = rng.multivariate_normal(mean0_6, C_ab)
        eta_bd = rng.multivariate_normal(mean0_6, C_bd)
        eta_ac = rng.multivariate_normal(mean0_6, C_ac)
        eta_cd = rng.multivariate_normal(mean0_6, C_cd)

        T_ab_s = exp_se3(eta_ab) @ T_ab
        T_bd_s = exp_se3(eta_bd) @ T_bd
        T_ac_s = exp_se3(eta_ac) @ T_ac
        T_cd_s = exp_se3(eta_cd) @ T_cd

        # B -> D
        T_bd_total = T_bd_s
        # C -> D
        T_cd_total = T_cd_s

        dp1 = rng.multivariate_normal(mean0_3, Cp)
        dp2 = rng.multivariate_normal(mean0_3, Cp)

        R_bd = T_bd_total[:3, :3]
        t_bd = T_bd_total[:3, 3]
        p1D = R_bd @ (p1B + dp1) + t_bd

        R_cd = T_cd_total[:3, :3]
        t_cd = T_cd_total[:3, 3]
        p2D = R_cd @ (p2C + dp2) + t_cd

        deltas[i, :] = (p2D - p1D)

    C_mc = cov_sample(deltas)

    err_ind = rel_frob(C_ind, C_mc)
    err_corr = rel_frob(C_corr, C_mc)

    print("\n=== Multi-edge correlation MC (BRANCHING) ===")
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
