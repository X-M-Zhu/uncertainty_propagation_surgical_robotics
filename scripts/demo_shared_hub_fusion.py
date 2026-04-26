# Author: X.M. Christine Zhu
# Date: 04/07/2026

"""
Demo: three paths through a shared hub — the combined topology.

Topology (from sketch):

         A                ← root (World frame)
         │
         B                ← shared uncertain hub (Optical Tracker)
        /│\\
       C D E              ← three marker branches
        \\│/
         F                ← goal frame (Tool tip — fused from all three paths)

Three paths from A to F, ALL sharing edge A→B:

    Path 1:  A → B → C → F
    Path 2:  A → B → D → F
    Path 3:  A → B → E → F

Surgical robotics analogy
--------------------------
    A  = World frame (operating room)
    B  = Optical tracker (registered to world, with registration uncertainty)
    C, D, E = Three tool marker spheres / reflective targets
    F  = Surgical tool tip (estimated by combining all three marker chains)

    All three marker chains pass through the SAME uncertain A→B registration.
    If the tracker shifts, ALL three markers shift together — their measurements
    are CORRELATED, not independent.

Key result
----------
    Naive fusion  (treats paths as independent):
        C_naive = inv(C_1^{-1} + C_2^{-1} + C_3^{-1})
        → counts A→B uncertainty THREE times instead of once
        → OVERCONFIDENT: reports a covariance that is too small

    Correct S-matrix fusion (Doc 2 unified framework):
        builds the full 18×18 stacked covariance S with off-diagonal blocks
        from the shared A→B edge, then applies the information form
        → gives a LARGER, honest covariance
        → confirmed by Monte Carlo

    Single path is correctly larger than both fused estimates —
    fusion still helps, just not as much as the naive formula claims.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from uncertainty_networks import GeometricNetwork, UncertainTransform
from uncertainty_networks.se3 import exp_se3, log_se3, inv_se3


# ── helpers ──────────────────────────────────────────────────────────────────

def make_edge(dx, dy, dz, sigma):
    """Pure translation edge with isotropic 6-DOF Gaussian uncertainty."""
    F = np.eye(4, dtype=float)
    F[:3, 3] = [dx, dy, dz]
    C = sigma**2 * np.eye(6, dtype=float)
    return UncertainTransform(F, C)


def sym(A):
    return 0.5 * (A + A.T)


def cov_sample(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


def frob_rel(A, B):
    return np.linalg.norm(A - B, "fro") / np.linalg.norm(B, "fro")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    seed    = 42
    N       = 50_000
    do_plot = True

    rng = np.random.default_rng(seed)

    print("=" * 65)
    print("  Shared-Hub Fusion — Three Paths Through a Common Edge A→B")
    print("=" * 65)
    print()
    print("Network topology:")
    print()
    print("       A          (root / World)")
    print("       │")
    print("       B          ← shared uncertain hub (Tracker)")
    print("      /│\\")
    print("     C D E        ← three marker branches")
    print("      \\│/")
    print("       F          ← goal / Tool tip  (three paths converge here)")
    print()
    print("  Path 1:  A → B → C → F")
    print("  Path 2:  A → B → D → F")
    print("  Path 3:  A → B → E → F")
    print("  Shared edge: A → B  (tracker registration — dominant uncertainty)")
    print()

    # ── build network ─────────────────────────────────────────────────────────
    #
    # Nominal geometry — all pure translations, nominally consistent:
    #
    #   A = (0,  0,  0)
    #   B = (0,  0,  0.30)   from A: +z  0.30
    #   C = (-0.40, 0,  0.60)  from B: -x 0.40, +z 0.30
    #   D = ( 0,  0.40, 0.50)  from B: +y 0.40, +z 0.20
    #   E = ( 0.40, 0,  0.60)  from B: +x 0.40, +z 0.30
    #   F = ( 0,  0,  1.00)   (all paths give same F_AF = translate(0,0,1.0) ✓)
    #       C → F:  +x 0.40, +z 0.40
    #       D → F:  -y 0.40, +z 0.50
    #       E → F:  -x 0.40, +z 0.40
    #
    # Uncertainty model (surgical tracker):
    #   σ_AB  = 0.08  m  (tracker-to-world registration — dominant)
    #   σ_mid = 0.04  m  (B → marker sphere)
    #   σ_last= 0.03  m  (marker sphere → tool tip)

    sigma_AB   = 0.08
    sigma_mid  = 0.04
    sigma_last = 0.03

    net = GeometricNetwork()

    # Shared hub
    net.add_edge("A", "B", make_edge( 0.00,  0.00,  0.30, sigma_AB))

    # Three branches from B  (to C, D, E)
    net.add_edge("B", "C", make_edge(-0.40,  0.00,  0.30, sigma_mid))
    net.add_edge("B", "D", make_edge( 0.00,  0.40,  0.20, sigma_mid))
    net.add_edge("B", "E", make_edge( 0.40,  0.00,  0.30, sigma_mid))

    # Three final links (all converging at F — nominally consistent)
    net.add_edge("C", "F", make_edge( 0.40,  0.00,  0.40, sigma_last))
    net.add_edge("D", "F", make_edge( 0.00, -0.40,  0.50, sigma_last))
    net.add_edge("E", "F", make_edge(-0.40,  0.00,  0.40, sigma_last))

    # ── analytic fusion via unified query_frame ────────────────────────────────
    result = net.query_frame("A", "F")

    print(f"  Paths found automatically: {result.n_paths}")
    print()

    C_paths, t_paths = [], []
    for i, pr in enumerate(result.path_results):
        C_i = pr.transform.C
        C_paths.append(C_i)
        t_paths.append(np.trace(C_i))
        print(f"  Path {i+1}:  {' → '.join(pr.path)}")
        print(f"    trace(C_{i+1}) = {t_paths[-1]:.6f}")
    print()

    C_correct = result.transform.C
    t_correct = np.trace(C_correct)
    print(f"  Correct S-matrix fusion:  trace(C_0) = {t_correct:.6f}")
    print()

    # ── naive fusion (independent assumption — WRONG) ─────────────────────────
    info_naive = sum(np.linalg.inv(sym(C)) for C in C_paths)
    C_naive    = np.linalg.inv(sym(info_naive))
    t_naive    = np.trace(C_naive)

    overconf = t_correct / t_naive          # > 1 means naive is overconfident
    print(f"  Naive fusion (independent paths):  trace(C_naive) = {t_naive:.6f}")
    print(f"  Naive is {overconf:.2f}× TOO CONFIDENT — underestimates uncertainty")
    print()
    print("  Root cause: A→B uncertainty is shared by all three paths.")
    print("  Naive counts it 3× instead of 1× → false confidence.")
    print()

    # ── build analytic S matrix for MVUE weights ──────────────────────────────
    # S_{ij} for paths i≠j = contribution from shared A→B edge only.
    # Since A→B is the FIRST edge in every path, the prefix adjoint is I,
    # so A_{AB}^i = I for all i, giving  S_{ij} = I · C_AB · I^T = C_AB.
    # S_{ii} = C_paths[i]  (standard per-path composition).

    C_AB = net.get_edge("A", "B").C
    m    = 3

    S_an = np.zeros((6*m, 6*m))
    for i in range(m):
        for j in range(m):
            if i == j:
                S_an[6*i:6*i+6, 6*j:6*j+6] = C_paths[i]
            else:
                S_an[6*i:6*i+6, 6*j:6*j+6] = C_AB   # shared A→B block
    S_an    = sym(S_an)
    S_inv   = np.linalg.inv(S_an)

    # Verify C_correct from S_an
    info_check = sum(S_inv[6*i:6*i+6, 6*j:6*j+6] for i in range(m) for j in range(m))
    C0_check   = np.linalg.inv(sym(info_check))
    assert frob_rel(C0_check, C_correct) < 1e-6, "S matrix mismatch — check edge geometry"
    print("  S-matrix analytic check passed ✓")
    print()

    # ── Monte Carlo validation ─────────────────────────────────────────────────
    print(f"Running Monte Carlo validation (N = {N:,}) ...")

    F_AF_nom = result.transform.F_nom
    F_AF_inv = inv_se3(F_AF_nom)

    F_AB = net.get_edge("A", "B").F_nom
    F_BC = net.get_edge("B", "C").F_nom
    F_BD = net.get_edge("B", "D").F_nom
    F_BE = net.get_edge("B", "E").F_nom
    F_CF = net.get_edge("C", "F").F_nom
    F_DF = net.get_edge("D", "F").F_nom
    F_EF = net.get_edge("E", "F").F_nom

    C_BC_e = net.get_edge("B", "C").C
    C_BD_e = net.get_edge("B", "D").C
    C_BE_e = net.get_edge("B", "E").C
    C_CF_e = net.get_edge("C", "F").C
    C_DF_e = net.get_edge("D", "F").C
    C_EF_e = net.get_edge("E", "F").C

    mean0       = np.zeros(6)
    xi_flat     = np.zeros((N, 6*m))

    for k in range(N):
        # Shared A→B: one draw for all three paths
        eta_AB = rng.multivariate_normal(mean0, C_AB)

        # Independent branch perturbations
        eta_BC = rng.multivariate_normal(mean0, C_BC_e)
        eta_BD = rng.multivariate_normal(mean0, C_BD_e)
        eta_BE = rng.multivariate_normal(mean0, C_BE_e)
        eta_CF = rng.multivariate_normal(mean0, C_CF_e)
        eta_DF = rng.multivariate_normal(mean0, C_DF_e)
        eta_EF = rng.multivariate_normal(mean0, C_EF_e)

        # Compose sampled paths
        T_AB_s = exp_se3(eta_AB) @ F_AB

        T1 = T_AB_s @ (exp_se3(eta_BC) @ F_BC) @ (exp_se3(eta_CF) @ F_CF)
        T2 = T_AB_s @ (exp_se3(eta_BD) @ F_BD) @ (exp_se3(eta_DF) @ F_DF)
        T3 = T_AB_s @ (exp_se3(eta_BE) @ F_BE) @ (exp_se3(eta_EF) @ F_EF)

        # Log-residuals at F
        xi_flat[k,  0: 6] = log_se3(T1 @ F_AF_inv)
        xi_flat[k,  6:12] = log_se3(T2 @ F_AF_inv)
        xi_flat[k, 12:18] = log_se3(T3 @ F_AF_inv)

    # Verify S matrix
    S_mc      = cov_sample(xi_flat)
    err_S_mc  = frob_rel(S_mc, S_an)
    print(f"  Stacked covariance S — MC vs analytic:  {err_S_mc:.4f}  ({err_S_mc*100:.1f}%)")

    # MVUE with correct weights: η_hat = C_0 · A_0^T S^{-1} y
    ones_block   = np.tile(np.eye(6), (1, m))       # 6 × 18: [I  I  I]
    A0T_Sinv     = ones_block @ S_inv                # 6 × 18

    xi_hat_corr  = (C_correct @ A0T_Sinv @ xi_flat.T).T   # N × 6
    C_mc_corr    = cov_sample(xi_hat_corr)
    err_corr     = frob_rel(C_mc_corr, C_correct)

    # Naive MVUE: η_hat = C_naive · (C_1^{-1} xi_1 + C_2^{-1} xi_2 + C_3^{-1} xi_3)
    blkdiag_Cinv = np.zeros((6*m, 6*m))
    for i in range(m):
        blkdiag_Cinv[6*i:6*i+6, 6*i:6*i+6] = np.linalg.inv(sym(C_paths[i]))
    A0T_Cinv_naive = ones_block @ blkdiag_Cinv                # 6 × 18

    xi_hat_naive = (C_naive @ A0T_Cinv_naive @ xi_flat.T).T  # N × 6
    C_mc_naive   = cov_sample(xi_hat_naive)
    err_naive    = frob_rel(C_mc_naive, C_naive)

    print(f"  Correct S-matrix estimator — MC error:  {err_corr:.4f}  ({err_corr*100:.1f}%)")
    print(f"  Naive estimator            — MC error:  {err_naive:.4f}  ({err_naive*100:.1f}%)  ← overconfident")
    print()

    np.set_printoptions(precision=6, suppress=True)
    print("  diag(C_correct) :", np.diag(C_correct))
    print("  diag(C_naive)   :", np.diag(C_naive))
    print("  diag(C_mc_corr) :", np.diag(C_mc_corr))

    # ── plot ──────────────────────────────────────────────────────────────────
    if do_plot:
        import matplotlib.pyplot as plt
        from uncertainty_networks.visualization import (
            plot_network_static,
            plot_network_interactive,
        )

        script_dir = os.path.dirname(os.path.abspath(__file__))

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        # Left: trace comparison — the key summary bar chart
        bar_labels = [
            "Single path\n(any one of 1-3)",
            f"Naive fusion\n(independent)\n← {overconf:.2f}× overconfident",
            "Correct fusion\n(S-matrix)\n← honest",
            "Monte Carlo\n(ground truth)",
        ]
        bar_values = [t_paths[0], t_naive, t_correct, np.trace(C_mc_corr)]
        bar_colors = ["#4878cf", "#d65f5f", "#2ca02c", "#9467bd"]

        bars = axes[0].bar(bar_labels, bar_values,
                           color=bar_colors, edgecolor="black", width=0.55)
        for bar, val in zip(bars, bar_values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=9,
            )
        axes[0].set_ylabel("trace(C)  —  total uncertainty")
        axes[0].set_title("Shared Hub: Naive vs Correct Fusion")
        axes[0].set_ylim(0, max(bar_values) * 1.35)
        axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")

        # Right: per-axis diagonal (rotation + translation)
        idx   = np.arange(6)
        w     = 0.25
        xlbls = [r"$\omega_x$", r"$\omega_y$", r"$\omega_z$",
                 r"$v_x$",      r"$v_y$",      r"$v_z$"]
        axes[1].bar(idx - w,   np.diag(C_naive),   width=w,
                    label=f"Naive (×{overconf:.2f} overconf.)",
                    color="#d65f5f", edgecolor="black")
        axes[1].bar(idx,       np.diag(C_correct), width=w,
                    label="Correct (S-matrix)",
                    color="#2ca02c", edgecolor="black")
        axes[1].bar(idx + w,   np.diag(C_mc_corr), width=w,
                    label="MC ground truth",
                    color="#9467bd", edgecolor="black")
        axes[1].set_xticks(idx)
        axes[1].set_xticklabels(xlbls, fontsize=10)
        axes[1].set_ylabel("variance per component")
        axes[1].set_title("Per-component: Naive vs Correct vs MC")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")

        plt.suptitle(
            "Shared Hub Fusion  (A→B→{C,D,E}→F)  |  "
            f"Shared edge A→B makes paths correlated\n"
            f"Naive: {err_naive*100:.1f}% MC error ({overconf:.2f}× overconfident)  |  "
            f"Correct: {err_corr*100:.1f}% MC error  |  "
            f"N = {N:,} samples",
            fontsize=9,
        )
        plt.tight_layout()

        # ── 3-D network visualisation ─────────────────────────────────────────
        _title_3d = (
            "Shared Hub Fusion — A→B→{C,D,E}→F\n"
            f"Shared edge A→B: naive is {overconf:.2f}× overconfident  |  "
            f"Correct MC error: {err_corr*100:.1f}%  (2σ ellipsoids)"
        )
        _frame_colors_static = {
            "A": "#2ca02c",   # green   — root
            "B": "#ff7f0e",   # orange  — shared hub (most uncertain)
            "C": "#1f77b4",   # blue    — branch 1
            "D": "#9467bd",   # purple  — branch 2
            "E": "#8c564b",   # brown   — branch 3
            "F": "#d62728",   # red     — goal
        }
        _frame_colors_inter = {
            "A": "#00FF99",   # bright green — root
            "B": "#FFEAA7",   # gold         — shared hub
            "C": "#4ECDC4",   # teal         — branch 1
            "D": "#45B7D1",   # sky blue     — branch 2
            "E": "#96CEB4",   # sage         — branch 3
            "F": "#FF6B6B",   # coral        — goal
        }

        plot_network_static(
            net, "A",
            title=_title_3d,
            ellipsoid_sigma=2,
            frame_colors=_frame_colors_static,
            save_path=os.path.join(script_dir, "shared_hub_3d.png"),
        )

        fig_3d = plot_network_interactive(
            net, "A",
            title=_title_3d,
            ellipsoid_sigma=2,
            frame_colors=_frame_colors_inter,
            save_path=os.path.join(script_dir, "shared_hub_3d.html"),
        )
        fig_3d.show()

        plt.show()


if __name__ == "__main__":
    main()
