# Author: X.M. Christine Zhu
# Date: 04/02/2026

"""
Demo and Monte Carlo validation for multi-path Bayesian fusion.

Scenario
--------
Two independent paths connect frame A to frame C, with the SAME nominal
transform (nominally consistent network):

         A
        / \\
       B   D
        \\ /
         C

    Path 1:  A --[+x]--> B --[+y]--> C    nominal total: [1, 1, 0]
    Path 2:  A --[+y]--> D --[+x]--> C    nominal total: [1, 1, 0]

Both paths reach the same C with the same nominal pose, so the fusion
formula is valid.

Each path independently propagates uncertainty and produces a Gaussian
estimate of the transform perturbation at C.  Since both paths estimate
the SAME quantity, Bayes' rule (information form) combines them optimally:

    C_fused  =  inv( C_1^{-1}  +  C_2^{-1} )

This gives a posterior covariance strictly smaller than either path alone.

Monte Carlo validation
----------------------
For each MC sample:
  1. Draw independent edge perturbations for all 4 edges.
  2. Compose each path to get xi_1, xi_2 (Log-residuals at C).
  3. Fuse: xi_fused = C_fused * (C_1^{-1} * xi_1 + C_2^{-1} * xi_2)
  4. Collect N samples of xi_fused.
  5. Sample covariance of xi_fused should match C_fused analytic.

Outputs
-------
- Per-path covariance diagonal and trace.
- Fused covariance diagonal and trace.
- Uncertainty reduction (%) vs best single path.
- Relative Frobenius error vs Monte Carlo (confirms fusion is correct).
- Bar chart comparing trace(C) across paths and fused result.
"""

import numpy as np

from uncertainty_networks import GeometricNetwork, UncertainTransform
from uncertainty_networks.se3 import exp_se3, log_se3, inv_se3


# ── helpers ──────────────────────────────────────────────────────────────────

def make_edge(translation, sigma):
    """Pure translation edge with isotropic Gaussian uncertainty."""
    F = np.eye(4, dtype=float)
    F[:3, 3] = translation
    C = sigma**2 * np.eye(6, dtype=float)
    return UncertainTransform(F, C)


def cov_sample(X: np.ndarray) -> np.ndarray:
    """Sample covariance of row-stacked data X, shape (N, d) -> (d, d)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


def frob_rel(A, B):
    """Relative Frobenius error ||A - B||_F / ||B||_F."""
    return np.linalg.norm(A - B, ord="fro") / np.linalg.norm(B, ord="fro")


def sym(A):
    return 0.5 * (A + A.T)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    seed    = 7
    N       = 30000
    sigma   = 0.1       # edge uncertainty (isotropic, same on all edges)
    do_plot = True

    rng = np.random.default_rng(seed)

    print("=" * 60)
    print("  Multi-Path Bayesian Fusion — Demo & MC Validation")
    print("=" * 60)
    print(f"Seed: {seed}    MC samples: {N}    sigma per edge: {sigma}")
    print()
    print("Network topology (nominally consistent diamond):")
    print()
    print("            A              ")
    print("           / \\           ")
    print("    [+x]  /   \\  [+y]   ")
    print("         B     D           ")
    print("    [+y]  \\   /  [+x]   ")
    print("           \\ /           ")
    print("            C              ")
    print()
    print("  Path 1:  A --[+x]--> B --[+y]--> C   nominal: [1, 1, 0]")
    print("  Path 2:  A --[+y]--> D --[+x]--> C   nominal: [1, 1, 0]")
    print("  Both paths reach the same nominal C  (nominally consistent)")
    print()

    # ── build network ────────────────────────────────────────────────────────
    net = GeometricNetwork()

    # Path 1: A --[1,0,0]--> B --[0,1,0]--> C   total: [1,1,0]
    net.add_edge("A", "B", make_edge([1.0, 0.0, 0.0], sigma))
    net.add_edge("B", "C", make_edge([0.0, 1.0, 0.0], sigma))

    # Path 2: A --[0,1,0]--> D --[1,0,0]--> C   total: [1,1,0]
    net.add_edge("A", "D", make_edge([0.0, 1.0, 0.0], sigma))
    net.add_edge("D", "C", make_edge([1.0, 0.0, 0.0], sigma))

    # ── analytic fusion via query_frame ──────────────────────────────────────
    result = net.query_frame("A", "C")

    print(f"Number of paths found: {result.n_paths}")
    print()

    C_list = []
    traces = []
    for i, pr in enumerate(result.path_results):
        t = np.trace(pr.transform.C)
        traces.append(t)
        C_list.append(pr.transform.C)
        print(f"  Path {i+1}: {' --> '.join(pr.path)}")
        print(f"    diag(C): {np.diag(pr.transform.C).round(5)}")
        print(f"    trace(C) = {t:.6f}")
        print()

    C_fused = result.transform.C
    t_fused = np.trace(C_fused)
    traces.append(t_fused)
    best_single = min(traces[:-1])
    reduction   = (1.0 - t_fused / best_single) * 100.0

    print(f"  Fused (Bayes' rule):  C_fused = inv( C_1^{{-1}} + C_2^{{-1}} )")
    print(f"    diag(C_fused): {np.diag(C_fused).round(5)}")
    print(f"    trace(C_fused) = {t_fused:.6f}")
    print(f"    Uncertainty reduction vs best single path: {reduction:.1f}%")
    print()

    # ── Monte Carlo validation ────────────────────────────────────────────────
    # For each sample:
    #   1. Draw edge perturbations independently.
    #   2. Compose each path -> xi_1, xi_2 (Log-residual at C).
    #   3. Fuse using MVUE weights: xi_f = C_fused*(C_1^{-1}*xi_1 + C_2^{-1}*xi_2)
    #   4. Cov(xi_f) should equal C_fused analytic.
    print("Running Monte Carlo validation ...")

    F_AC_nom = result.transform.F_nom
    F_AC_inv = inv_se3(F_AC_nom)

    # Nominal transforms per edge
    F_AB = net.get_edge("A", "B").F_nom
    F_BC = net.get_edge("B", "C").F_nom
    F_AD = net.get_edge("A", "D").F_nom
    F_DC = net.get_edge("D", "C").F_nom

    # Covariances per edge
    C_AB = net.get_edge("A", "B").C
    C_BC = net.get_edge("B", "C").C
    C_AD = net.get_edge("A", "D").C
    C_DC = net.get_edge("D", "C").C

    # Precision matrices (per path and fused)
    C1 = C_list[0]
    C2 = C_list[1]
    C1_inv = np.linalg.inv(sym(C1))
    C2_inv = np.linalg.inv(sym(C2))
    Cf_inv = np.linalg.inv(sym(C_fused))

    mean0 = np.zeros(6, dtype=float)
    xi_fused_samples = np.zeros((N, 6), dtype=float)

    for i in range(N):
        # Sample edge perturbations
        eta_AB = rng.multivariate_normal(mean0, C_AB)
        eta_BC = rng.multivariate_normal(mean0, C_BC)
        eta_AD = rng.multivariate_normal(mean0, C_AD)
        eta_DC = rng.multivariate_normal(mean0, C_DC)

        # Compose path 1: T1 = Exp(eta_AB)*F_AB @ Exp(eta_BC)*F_BC
        T1 = (exp_se3(eta_AB) @ F_AB) @ (exp_se3(eta_BC) @ F_BC)
        xi1 = log_se3(T1 @ F_AC_inv)

        # Compose path 2: T2 = Exp(eta_AD)*F_AD @ Exp(eta_DC)*F_DC
        T2 = (exp_se3(eta_AD) @ F_AD) @ (exp_se3(eta_DC) @ F_DC)
        xi2 = log_se3(T2 @ F_AC_inv)

        # Fuse: MVUE = C_fused * (C_1^{-1} * xi_1 + C_2^{-1} * xi_2)
        xi_fused_samples[i] = C_fused @ (C1_inv @ xi1 + C2_inv @ xi2)

    C_mc = cov_sample(xi_fused_samples)
    err  = frob_rel(C_mc, C_fused)

    print(f"Relative Frobenius error (analytic vs MC): {err:.4f}")
    print()
    np.set_printoptions(precision=6, suppress=True)
    print("diag(C_fused analytic):", np.diag(C_fused))
    print("diag(C_fused MC)      :", np.diag(C_mc))

    # ── plot ─────────────────────────────────────────────────────────────────
    if do_plot:
        import os
        import matplotlib.pyplot as plt
        from uncertainty_networks.visualization import (
            plot_network_static,
            plot_network_interactive,
        )

        script_dir = os.path.dirname(os.path.abspath(__file__))

        labels = [f"Path {i+1}\n{' -> '.join(pr.path)}"
                  for i, pr in enumerate(result.path_results)] + ["Fused\n(Bayes' rule)"]
        colors = ["steelblue"] * len(result.path_results) + ["tomato"]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, traces, color=colors, edgecolor="black", width=0.5)

        for bar, val in zip(bars, traces):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{val:.4f}",
                    ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("trace(C)  —  total uncertainty")
        ax.set_title(
            "Multi-Path Bayesian Fusion: uncertainty per path vs fused result\n"
            f"(sigma={sigma} per edge,  {N} MC samples,  MC error={err:.3f})"
        )
        ax.set_ylim(0, max(traces) * 1.2)
        plt.tight_layout()

        # ── 3-D network visualisation ─────────────────────────────────────────
        # sigma=0.1 per edge → 2σ radius = 0.2 m in a 1 m network.
        # Ellipsoids are naturally visible at sigma=2.
        # At C the fused ellipsoid (query_frame) is shown — smaller than
        # either single-path ellipsoid, illustrating the fusion benefit.
        _title_3d = (
            "Multi-Path Fusion — Diamond Network\n"
            "Fused covariance at C is smaller than either single path  (2σ ellipsoids)"
        )

        # Static — for poster / report
        plot_network_static(
            net, "A",
            title=_title_3d,
            ellipsoid_sigma=2,
            frame_colors={"A": "#2ca02c", "B": "#1f77b4", "C": "#d62728", "D": "#9467bd"},
            save_path=os.path.join(script_dir, "multipath_3d.png"),
        )

        # Interactive — for live demo at poster session
        fig_3d = plot_network_interactive(
            net, "A",
            title=_title_3d,
            ellipsoid_sigma=2,
            frame_colors={"A": "#2CA02C", "B": "#1F77B4", "C": "#D62728", "D": "#9467BD"},
            save_path=os.path.join(script_dir, "multipath_3d.html"),
        )
        fig_3d.show()

        plt.show()


if __name__ == "__main__":
    main()
