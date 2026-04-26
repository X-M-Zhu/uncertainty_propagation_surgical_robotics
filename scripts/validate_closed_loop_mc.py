# Author: X.M. Christine Zhu
# Date: 02/08/2026

"""
Monte Carlo validation for closed-loop conditioning (loop residual constraint).

Validate the posterior covariance computed by:
    C_post = (C0^{-1} + H^T C_nu^{-1} H)^{-1}

against a Monte Carlo approximation of the same constraint.

Model
-----
Prior:
    eta_res ~ N(0, C_res)
    eta_k   ~ N(0, C_k)
independent, so x = [eta_res; eta_k] ~ N(0, C0)

Residual (nonlinear):
    r = Log( (Exp(eta_res)F_res)^{-1} (Exp(eta_k)F_k) )

Constraint (observation):
    r ≈ 0 with small noise

Monte Carlo posterior approximation
-----------------------------------
Generate many prior samples and accept/reweight samples that satisfy the
constraint approximately. Two options:

(A) REJECTION (default):
    accept if ||r|| <= tau
This approximates conditioning on r=0 with small noise.

(B) WEIGHTING (optional):
    weight w_i ∝ exp(-0.5 r_i^T C_nu^{-1} r_i)
This approximates Bayesian conditioning more smoothly.

Then estimate posterior covariance of eta_res and compare to analytic.

Notes
-----
- Keep uncertainties small for first-order assumptions.
- Rejection threshold tau may need tuning; script prints acceptance rate.
"""

from __future__ import annotations

import numpy as np

from uncertainty_networks.se3 import make_se3, rotz
from uncertainty_networks.closed_loop import (
    linearize_loop_residual,
    condition_on_loop,
    loop_residual,
)


def cov_sample(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


def cov_weighted(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Weighted covariance of row-stacked data X (N,d) with weights w (N,).
    Uses normalized weights. Returns (d,d).
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    w = w / (w.sum() + 1e-18)
    mu = (w[:, None] * X).sum(axis=0)
    Xc = X - mu[None, :]
    # second moment
    C = (w[:, None] * Xc).T @ Xc
    return 0.5 * (C + C.T)


def rel_frob(A: np.ndarray, B: np.ndarray, eps: float = 1e-18) -> float:
    denom = float(np.linalg.norm(B, ord="fro"))
    if denom < eps:
        return float(np.linalg.norm(A - B, ord="fro"))
    return float(np.linalg.norm(A - B, ord="fro") / denom)


def main():
    # Configuration
    seed = 10
    rng = np.random.default_rng(seed)

    N_prior = 250000  # number of prior samples
    method = "weighting"  # "rejection" or "weighting"

    # Rejection tuning (if method == "rejection")
    tau = 6e-3  # residual norm threshold (tune if acceptance too low/high)

    # Weighting tuning (if method == "weighting")
    C_nu = 1e-3 * np.eye(6)  # observation noise (smaller => stronger conditioning)

    # Nominal transforms
    F_res = make_se3(rotz(0.12), [0.10, 0.02, -0.01])
    F_k   = make_se3(rotz(0.08), [0.08, -0.01, 0.03])

    # Prior covariances (small)
    C_res = np.diag([6e-6, 6e-6, 6e-6, 8e-6, 8e-6, 8e-6])
    C_k   = np.diag([5e-6, 5e-6, 5e-6, 7e-6, 7e-6, 7e-6])

    # Analytic posterior
    lin = linearize_loop_residual(F_res, F_k)
    post = condition_on_loop(C_res, C_k, lin, C_nu=C_nu)

    C_res_analytic = post.C_res
    C_k_analytic = post.C_k

    # Monte Carlo posterior approx
    mean0 = np.zeros(6)
    eta_res_samples = rng.multivariate_normal(mean0, C_res, size=N_prior)
    eta_k_samples = rng.multivariate_normal(mean0, C_k, size=N_prior)

    # compute residuals
    r_vals = np.zeros((N_prior, 6), dtype=float)
    for i in range(N_prior):
        r_vals[i, :] = loop_residual(F_res, F_k, eta_res_samples[i], eta_k_samples[i])

    if method == "rejection":
        r_norm = np.linalg.norm(r_vals, axis=1)
        mask = r_norm <= tau
        accepted = int(mask.sum())
        rate = accepted / N_prior

        if accepted < 2000:
            print(f"[WARN] acceptance too low: {accepted}/{N_prior} ({rate:.4%}). Increase tau.")
        if accepted > 0.3 * N_prior:
            print(f"[WARN] acceptance high: {accepted}/{N_prior} ({rate:.4%}). Decrease tau for tighter constraint.")

        eta_res_post = eta_res_samples[mask, :]
        eta_k_post = eta_k_samples[mask, :]

        C_res_mc = cov_sample(eta_res_post)
        C_k_mc = cov_sample(eta_k_post)

        print("\n=== Closed-loop MC validation (REJECTION) ===")
        print(f"Seed: {seed}")
        print(f"N_prior: {N_prior}")
        print(f"tau: {tau}")
        print(f"Accepted: {accepted} ({rate:.4%})")

    elif method == "weighting":
    # Auto-tune scalar residual noise: C_nu = s * I
    # Goal: keep effective sample size reasonably large to avoid weight collapse.
        target_eff = 1500.0
        s = float(np.diag(C_nu)[0])

        for _ in range(20):
            C_nu_try = s * np.eye(6)
            Cnu_inv = np.linalg.inv(C_nu_try)

            quad = np.einsum("ni,ij,nj->n", r_vals, Cnu_inv, r_vals)
            quad = quad - quad.min()
            w = np.exp(-0.5 * quad)

            eff = (w.sum() ** 2) / (np.sum(w ** 2) + 1e-18)

            if eff < target_eff:
                s *= 3.0   # loosen faster but not crazy
                continue

            # if we reached target_eff, stop
            break

        C_res_mc = cov_weighted(eta_res_samples, w)
        C_k_mc = cov_weighted(eta_k_samples, w)

        print(f"Auto-tuned scalar C_nu: s={s:.3e}")
        print("\n=== Closed-loop MC validation (WEIGHTING) ===")
        print(f"Seed: {seed}")
        print(f"N_prior: {N_prior}")
        print(f"Effective sample size (approx): {eff:.1f}")

    else:
        raise ValueError("method must be 'rejection' or 'weighting'")

    # Compare analytic vs MC
    err_res = rel_frob(C_res_analytic, C_res_mc)
    err_k = rel_frob(C_k_analytic, C_k_mc)

    print("")
    print(f"rel frob error cov(eta_res): {err_res:.4f}")
    print(f"rel frob error cov(eta_k)  : {err_k:.4f}")
    print("")
    print(f"trace prior res: {np.trace(C_res):.6e}   trace post analytic res: {np.trace(C_res_analytic):.6e}   trace post MC res: {np.trace(C_res_mc):.6e}")
    print(f"trace prior k  : {np.trace(C_k):.6e}     trace post analytic k  : {np.trace(C_k_analytic):.6e}     trace post MC k  : {np.trace(C_k_mc):.6e}")

    np.set_printoptions(precision=6, suppress=False)
    print("\ndiag(C_res analytic):", np.diag(C_res_analytic))
    print("diag(C_res mc)      :", np.diag(C_res_mc))


if __name__ == "__main__":
    main()
