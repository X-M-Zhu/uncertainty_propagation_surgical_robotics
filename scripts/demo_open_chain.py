# Author: X.M. Christine Zhu
# Date: 04/02/2026

"""
Demo: uncertainty propagation along an open kinematic chain.

Scenario
--------
A simple robot arm with three links, modelled as a directed chain of
uncertain SE(3) transforms:

    World --> Shoulder --> Elbow --> Tool

Each edge stores a nominal rigid transform (position + orientation) and a
6x6 covariance matrix describing how uncertain that transform is.

The covariance accumulates along the chain via the adjoint mapping:

    C_AC  =  C_AB  +  Ad_{F_nom,AB} * C_BC * Ad_{F_nom,AB}^T

This is the first-order CIS I left-perturbation propagation rule.

Outputs
-------
- Nominal position of each frame in the World frame.
- Propagated covariance diagonal and trace at each frame.
- How much each link contributes to the total uncertainty at the Tool.
- A bar chart showing how uncertainty grows along the chain.
"""

import numpy as np

from uncertainty_networks import GeometricNetwork, UncertainTransform
from uncertainty_networks.se3 import make_se3, rotz


# ── helpers ──────────────────────────────────────────────────────────────────

def make_edge(translation, rot_deg, sigma_rot, sigma_trans):
    """
    Build an edge with a rotation about Z and a translation.

    Parameters
    ----------
    translation : list[float]   nominal translation [x, y, z] in metres
    rot_deg     : float         nominal rotation about Z axis in degrees
    sigma_rot   : float         1-sigma rotation uncertainty in radians
    sigma_trans : float         1-sigma translation uncertainty in metres
    """
    R = rotz(np.deg2rad(rot_deg))
    F = make_se3(R, translation)
    C = np.diag([sigma_rot**2]   * 3 +
                [sigma_trans**2] * 3)
    return UncertainTransform(F, C)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    do_plot = True

    print("=" * 55)
    print("  Open-Chain Uncertainty Propagation Demo")
    print("=" * 55)
    print()
    print("Chain:  World --> Shoulder --> Elbow --> Tool")
    print()

    # ── build network ────────────────────────────────────────────────────────
    net = GeometricNetwork()

    #                       translation      rot    sigma_rot   sigma_trans
    net.add_edge("World",    "Shoulder", make_edge([0.0, 0.0, 0.5],  0.0,  0.002, 0.001))
    net.add_edge("Shoulder", "Elbow",   make_edge([0.3, 0.0, 0.0], 15.0,  0.003, 0.002))
    net.add_edge("Elbow",    "Tool",    make_edge([0.2, 0.0, 0.0], -8.0,  0.002, 0.001))

    # ── query each frame from World ──────────────────────────────────────────
    frames  = ["Shoulder", "Elbow", "Tool"]
    results = {f: net.query("World", f) for f in frames}

    print(f"{'Frame':<12}  {'Position (x,y,z) m':^32}  {'trace(C)':>10}  {'std rot (mrad)':>15}  {'std trans (mm)':>14}")
    print("-" * 90)

    for f in frames:
        r   = results[f]
        pos = r.transform.F_nom[:3, 3]
        C   = r.transform.C
        std_rot   = np.sqrt(np.mean(np.diag(C)[:3])) * 1e3   # mrad
        std_trans = np.sqrt(np.mean(np.diag(C)[3:])) * 1e3   # mm
        print(f"{f:<12}  [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]  "
              f"{np.trace(C):>10.6f}  "
              f"{std_rot:>15.4f}  "
              f"{std_trans:>14.4f}")

    # ── per-edge contribution at Tool ────────────────────────────────────────
    print()
    print("Uncertainty contribution per edge at Tool:")
    print()

    r_tool = results["Tool"]
    C_tool = r_tool.transform.C

    # Contributions: difference in trace when each edge is removed
    # We approximate by querying sub-chains and using additivity of trace.
    # More precisely: trace contribution of edge i = trace(Ad_prefix * C_i * Ad_prefix^T)
    from uncertainty_networks.se3 import adjoint_se3

    def edge_contribution(net, chain, edge_idx):
        """Trace contribution of edge at edge_idx to the end of chain."""
        # Build prefix transform up to (but not including) this edge
        prefix = np.eye(4)
        for a, b in chain[:edge_idx]:
            e = net.get_edge(a, b)
            prefix = prefix @ e.F_nom
        Ad = adjoint_se3(prefix)
        e = net.get_edge(*chain[edge_idx])
        C_e = e.C
        C_mapped = Ad @ C_e @ Ad.T
        return np.trace(C_mapped), C_mapped

    chain = [("World", "Shoulder"), ("Shoulder", "Elbow"), ("Elbow", "Tool")]
    labels = ["World → Shoulder", "Shoulder → Elbow", "Elbow → Tool"]

    contribs = []
    for i, label in enumerate(labels):
        t, _ = edge_contribution(net, chain, i)
        contribs.append(t)
        pct = 100.0 * t / np.trace(C_tool)
        print(f"  {label:<22}  trace contribution = {t:.6f}  ({pct:.1f}%)")

    print()
    print(f"  Total trace(C_tool) = {np.trace(C_tool):.6f}")
    print()
    print("Key observations:")
    print("  - Uncertainty grows at every link (covariance only accumulates).")
    print("  - Edges closer to the base (World) are mapped by larger adjoints")
    print("    and tend to contribute more to the final uncertainty at Tool.")

    # ── plot ─────────────────────────────────────────────────────────────────
    if do_plot:
        import os
        import matplotlib.pyplot as plt
        from uncertainty_networks.visualization import (
            plot_network_static,
            plot_network_interactive,
        )

        script_dir = os.path.dirname(os.path.abspath(__file__))

        _, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: trace(C) growing along chain
        chain_frames = ["World", "Shoulder", "Elbow", "Tool"]
        traces = [0.0] + [np.trace(results[f].transform.C) for f in frames]

        axes[0].plot(chain_frames, traces, marker="o", color="steelblue", linewidth=2)
        for i, (fr, tr) in enumerate(zip(chain_frames, traces)):
            axes[0].annotate(f"{tr:.5f}", (fr, tr),
                             textcoords="offset points", xytext=(0, 8),
                             ha="center", fontsize=8)
        axes[0].set_ylabel("trace(C)  —  total uncertainty")
        axes[0].set_title("Uncertainty growth along the chain")
        axes[0].set_ylim(bottom=0)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Right: per-edge contribution at Tool
        colors = ["#4878cf", "#6acc65", "#d65f5f"]
        bars = axes[1].bar(labels, contribs, color=colors, edgecolor="black", width=0.5)
        for bar, val in zip(bars, contribs):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.0001,
                         f"{val:.5f}",
                         ha="center", va="bottom", fontsize=8)
        axes[1].set_ylabel("trace contribution at Tool")
        axes[1].set_title("Per-edge uncertainty contribution at Tool")
        axes[1].set_ylim(0, max(contribs) * 1.25)
        axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")

        plt.suptitle("Open-Chain Uncertainty Propagation — World → Shoulder → Elbow → Tool",
                     fontsize=11)
        plt.tight_layout()

        # ── 3-D network visualisation ─────────────────────────────────────────
        # ellipsoid_sigma=25: physical uncertainties are ~1-3 mm in a ~500 mm
        # network, so a large sigma is used to make ellipsoids visible on poster.
        _title_3d = (
            "Open Chain: Uncertainty Propagation\n"
            "World → Shoulder → Elbow → Tool  (25σ ellipsoids)"
        )

        # Static — for poster / report
        plot_network_static(
            net, "World",
            title=_title_3d,
            ellipsoid_sigma=25,
            save_path=os.path.join(script_dir, "open_chain_3d.png"),
        )

        # Interactive — for live demo at poster session
        fig_3d = plot_network_interactive(
            net, "World",
            title=_title_3d,
            ellipsoid_sigma=25,
            save_path=os.path.join(script_dir, "open_chain_3d.html"),
        )
        fig_3d.show()

        plt.show()


if __name__ == "__main__":
    main()
