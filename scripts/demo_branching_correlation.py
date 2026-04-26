# Author: X.M. Christine Zhu
# Date: 04/02/2026

"""
Demo: branching paths with shared-edge correlation.

Scenario
--------
Two paths branch from a shared hub frame B:

    A --> B --> C      (e.g. World -> Tracker -> Tool tip)
    A --> B --> D      (e.g. World -> Tracker -> Anatomical landmark)

The edge A --> B is SHARED by both paths.
C and D have DIFFERENT destinations — this is NOT multi-path fusion.

The key insight:
    Because both paths pass through the same uncertain edge A --> B,
    the uncertainty at C and the uncertainty at D are CORRELATED.
    If B shifts due to its uncertain registration to A, both C and D
    shift together.

Naive approach (wrong):
    Assume C and D are independent.
    C_delta = C_C + C_D              <- overestimates uncertainty

Correlation-aware approach (correct):
    Track the shared edge A --> B.
    C_delta = C_C + C_D - Cross - Cross^T

    where Cross = Cov(p_C, p_D) = sum over shared edges e:
                                       J_C,e * C_e * J_D,e^T

The naive estimate can be several times too large.
The correlation-aware estimate matches Monte Carlo.

Surgical robotics analogy:
    A = World (operating room)
    B = Optical tracker (registered to world with some uncertainty)
    C = Tool tip (tracked by the tracker)
    D = Anatomical landmark (tracked by the same tracker)

    Both C and D are measured via the same tracker.
    If the tracker registration (A->B) is uncertain, both move together.
    Their RELATIVE distance is therefore more certain than the naive
    sum of individual uncertainties would suggest.

Outputs
-------
- Individual covariance at C and D.
- Naive vs correlation-aware C_delta for the relative vector C->D.
- How much the naive estimate overestimates uncertainty.
- Relative Frobenius error vs Monte Carlo for both methods.
- Bar chart comparing trace(C_delta) for naive, correct, and MC.
"""

import numpy as np

from uncertainty_networks import GeometricNetwork, UncertainTransform
from uncertainty_networks.se3 import make_se3, rotz, exp_se3


# ── helpers ──────────────────────────────────────────────────────────────────

def make_edge(translation, rot_deg, C_diag):
    """Edge with rotation about Z, translation, and diagonal covariance."""
    R = rotz(np.deg2rad(rot_deg))
    F = make_se3(R, translation)
    C = np.diag(C_diag)
    return UncertainTransform(F, C)


def cov_sample(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / (X.shape[0] - 1)


def frob_rel(A, B):
    return np.linalg.norm(A - B, ord="fro") / np.linalg.norm(B, ord="fro")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    seed    = 6
    N       = 60000
    do_plot = True

    rng = np.random.default_rng(seed)

    print("=" * 60)
    print("  Branching Paths — Shared-Edge Correlation Demo")
    print("=" * 60)
    print()
    print("Network topology:")
    print()
    print("    A (World)")
    print("    |")
    print("    B (Tracker)    <-- shared uncertain edge A->B")
    print("   / \\")
    print("  C   D")
    print("(Tool) (Landmark)")
    print()
    print("  Path 1:  A --> B --> C   (World -> Tracker -> Tool tip)")
    print("  Path 2:  A --> B --> D   (World -> Tracker -> Landmark)")
    print("  Shared edge: A --> B  (tracker registration to world)")
    print()

    # ── build network ────────────────────────────────────────────────────────
    # Use small sigma values (realistic surgical robotics scale) so that
    # the first-order Gaussian approximation is valid.
    #
    # sigma_rot   ~ 1-3 mrad  (well-calibrated optical tracker)
    # sigma_trans ~ 1-2 mm
    #
    # A->B has the LARGEST uncertainty (tracker-to-world registration).
    # This is what creates the strong correlation between C and D.

    C_AB = np.diag([4e-6, 4e-6, 4e-6,  4e-6, 4e-6, 4e-6])   # tracker reg.
    C_BC = np.diag([1e-6, 1e-6, 1e-6,  1e-6, 1e-6, 1e-6])   # tool tracking
    C_BD = np.diag([1e-6, 1e-6, 1e-6,  1e-6, 1e-6, 1e-6])   # landmark tracking

    net = GeometricNetwork()

    # Shared: World -> Tracker
    T_AB = make_se3(rotz(np.deg2rad(5.0)), [0.10, 0.00, 0.00])
    net.add_edge("A", "B", UncertainTransform(T_AB, C_AB))

    # Branch 1: Tracker -> Tool tip
    T_BC = make_se3(rotz(np.deg2rad(2.0)), [0.10, 0.00, 0.00])
    net.add_edge("B", "C", UncertainTransform(T_BC, C_BC))

    # Branch 2: Tracker -> Anatomical landmark
    T_BD = make_se3(rotz(np.deg2rad(-3.0)), [0.00, 0.10, 0.00])
    net.add_edge("B", "D", UncertainTransform(T_BD, C_BD))

    # Attach points
    Cp  = 1e-7 * np.eye(3)
    p_C = np.array([0.02,  0.00,  0.00])
    p_D = np.array([0.00, -0.02,  0.00])

    net.add_point("tool_tip",  "C", p_C, Cp)
    net.add_point("landmark",  "D", p_D, Cp)

    # ── individual frame covariances ─────────────────────────────────────────
    r_C = net.query("A", "C")
    r_D = net.query("A", "D")

    print("Individual frame covariances (queried from A):")
    print(f"  trace(C at C) = {np.trace(r_C.transform.C):.3e}  "
          f"path: {' -> '.join(r_C.path)}")
    print(f"  trace(C at D) = {np.trace(r_D.transform.C):.3e}  "
          f"path: {' -> '.join(r_D.path)}")
    print()

    # ── relative vector: naive vs correlation-aware ──────────────────────────
    _, C_ind  = net.query_relative_vector_independent("tool_tip", "landmark", "A")
    delta_corr, C_corr = net.query_relative_vector("tool_tip", "landmark", "A")

    print("Relative vector (tool_tip -> landmark) in frame A:")
    print(f"  delta = [{delta_corr[0]:+.4f}, {delta_corr[1]:+.4f}, {delta_corr[2]:+.4f}] m")
    print()
    print(f"  Naive  (independent) trace(C_delta) = {np.trace(C_ind):.3e}")
    print(f"  Correct (corr-aware) trace(C_delta) = {np.trace(C_corr):.3e}")
    overest = np.trace(C_ind) / np.trace(C_corr)
    print(f"  Naive overestimates by factor: {overest:.2f}x")
    print()
    print("  Why? The shared A->B edge moves BOTH points together.")
    print("  Their relative distance is far less affected by tracker")
    print("  uncertainty than their absolute positions individually.")
    print()

    # ── Monte Carlo validation ────────────────────────────────────────────────
    print("Running Monte Carlo validation ...")

    mean0_6 = np.zeros(6)
    mean0_3 = np.zeros(3)
    deltas_mc = np.zeros((N, 3))

    for i in range(N):
        # Sample edge perturbations (shared A->B drawn once for both paths)
        eta_AB = rng.multivariate_normal(mean0_6, C_AB)
        eta_BC = rng.multivariate_normal(mean0_6, C_BC)
        eta_BD = rng.multivariate_normal(mean0_6, C_BD)

        # Compose paths (same perturbed T_AB used for both)
        T_AB_s = exp_se3(eta_AB) @ T_AB
        T_AC_s = T_AB_s @ (exp_se3(eta_BC) @ T_BC)   # A -> B -> C
        T_AD_s = T_AB_s @ (exp_se3(eta_BD) @ T_BD)   # A -> B -> D

        # Sample local point noise
        dp_C = rng.multivariate_normal(mean0_3, Cp)
        dp_D = rng.multivariate_normal(mean0_3, Cp)

        # Express points in frame A
        p_C_in_A = T_AC_s[:3, :3] @ (p_C + dp_C) + T_AC_s[:3, 3]
        p_D_in_A = T_AD_s[:3, :3] @ (p_D + dp_D) + T_AD_s[:3, 3]

        deltas_mc[i] = p_D_in_A - p_C_in_A

    C_mc = cov_sample(deltas_mc)

    err_ind  = frob_rel(C_ind,  C_mc)
    err_corr = frob_rel(C_corr, C_mc)

    print(f"  Relative Frobenius error — naive:        {err_ind:.4f}  ({err_ind*100:.1f}%)")
    print(f"  Relative Frobenius error — corr-aware:   {err_corr:.4f}  ({err_corr*100:.1f}%)")
    print()
    np.set_printoptions(precision=6, suppress=True)
    print("  diag(C_mc)   :", np.diag(C_mc))
    print("  diag(C_ind)  :", np.diag(C_ind))
    print("  diag(C_corr) :", np.diag(C_corr))

    # ── plot ─────────────────────────────────────────────────────────────────
    if do_plot:
        import os
        import matplotlib.pyplot as plt
        from uncertainty_networks.visualization import (
            plot_network_static,
            plot_network_interactive,
        )

        script_dir = os.path.dirname(os.path.abspath(__file__))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: trace comparison
        methods = ["Naive\n(independent)", "Corr-aware\n(correct)", "Monte Carlo\n(ground truth)"]
        values  = [np.trace(C_ind), np.trace(C_corr), np.trace(C_mc)]
        colors  = ["#d65f5f", "#4878cf", "#6acc65"]

        bars = axes[0].bar(methods, values, color=colors, edgecolor="black", width=0.5)
        for bar, val in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() * 1.02,
                         f"{val:.2e}",
                         ha="center", va="bottom", fontsize=9)
        axes[0].set_ylabel("trace(C_delta)  —  total relative uncertainty")
        axes[0].set_title("Naive vs Correlation-Aware vs Monte Carlo")
        axes[0].set_ylim(0, max(values) * 1.25)
        axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")

        # Right: per-axis diagonal comparison
        idx = np.arange(3)
        w   = 0.25
        axes[1].bar(idx - w, np.diag(C_ind),  width=w, label="Naive",
                    color="#d65f5f", edgecolor="black")
        axes[1].bar(idx,     np.diag(C_corr), width=w, label="Corr-aware",
                    color="#4878cf", edgecolor="black")
        axes[1].bar(idx + w, np.diag(C_mc),   width=w, label="Monte Carlo",
                    color="#6acc65", edgecolor="black")
        axes[1].set_xticks(idx)
        axes[1].set_xticklabels(["x", "y", "z"])
        axes[1].set_ylabel("variance per axis")
        axes[1].set_title("Per-axis variance: Naive vs Correct vs MC")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")

        plt.suptitle(
            "Branching Paths — Shared Edge A→B creates correlation between C and D\n"
            f"(N={N} MC samples,  naive error={err_ind:.3f},  corr-aware error={err_corr:.3f})",
            fontsize=10
        )
        plt.tight_layout()

        # ── 3-D network visualisation ─────────────────────────────────────────
        # ellipsoid_sigma=8: uncertainties are ~3 mm in a ~200 mm network.
        # Shared edge A->B drives most of the uncertainty at C and D.
        _title_3d = (
            "Branching Paths — Shared Edge A→B\n"
            "Correlation between C (tool tip) and D (landmark)  (8σ ellipsoids)"
        )

        # Static — for poster / report
        plot_network_static(
            net, "A",
            title=_title_3d,
            ellipsoid_sigma=8,
            frame_colors={"A": "#2ca02c", "B": "#ff7f0e", "C": "#1f77b4", "D": "#d62728"},
            save_path=os.path.join(script_dir, "branching_3d.png"),
        )

        # Interactive — for live demo at poster session
        fig_3d = plot_network_interactive(
            net, "A",
            title=_title_3d,
            ellipsoid_sigma=8,
            frame_colors={"A": "#2CA02C", "B": "#FF7F0E", "C": "#1F77B4", "D": "#D62728"},
            save_path=os.path.join(script_dir, "branching_3d.html"),
        )
        fig_3d.show()

        plt.show()


if __name__ == "__main__":
    main()
