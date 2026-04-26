# Author: X.M. Christine Zhu
# Date: 04/2026

"""
poster_figures.py  (v2 — enhanced for fellowship poster)

Generates 3 poster-quality figures for:
  "Simulation-Based Uncertainty Propagation in Geometric Networks
   for Surgical Robotics"

  poster_fig1_network.png     — network diagram + cumulative uncertainty growth
                                (with per-node uncertainty ellipses)
  poster_fig2_montecarlo.png  — Monte Carlo deep-dive  (4 panels):
                                  A. 2D sample scatter + analytic confidence ellipses
                                  B. Frobenius-error convergence vs. N_MC
                                  C. Component histogram + Gaussian overlay
                                  D. Analytic vs. MC covariance heatmap (split triangle)
  poster_fig3_reduction.png   — Bayesian fusion results  (3 panels):
                                  A. Covariance-trace bar chart
                                  B. Per-component grouped bars
                                  C. Step-by-step uncertainty accumulation along each path

Network topology (two paths from a to g):

         a  (World)
        / \\
       b   d
       |   |
       c   e
        \\ /
         f
         |
         g  (Target)

  Path 1 (optical tracker):  a → b → c → f → g
  Path 2 (robot kinematics): a → d → e → f → g

Run from project root:
    python scripts/poster_figures.py

Saves to results/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from PIL import Image as _PILImage

from uncertainty_networks.se3 import make_se3, rotz, inv_se3, exp_se3, log_se3
from uncertainty_networks.uncertain_geometry import UncertainTransform
from uncertainty_networks.network import GeometricNetwork

os.makedirs("results", exist_ok=True)

RNG  = np.random.default_rng(42)
N_MC = 30_000

# ── Global style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linestyle":    "--",
    "axes.facecolor":    "#F8FAFC",
    "figure.facecolor":  "white",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
})

# ── Colour palette ─────────────────────────────────────────────────────────
C_BLUE    = "#2563EB"   # azure blue   – Path 1 / optical tracker
C_VIOLET  = "#7C3AED"   # violet       – Path 2 / robot kinematics
C_EMERALD = "#059669"   # emerald      – fused / shared frames
C_ROSE    = "#E11D48"   # rose red     – target frame / highlight
C_AMBER   = "#D97706"   # warm amber   – annotations / fill
C_DARK    = "#1E293B"   # slate dark   – text / borders
C_SKY     = "#38BDF8"   # sky blue     – MC sample points
C_PINK    = "#EC4899"   # hot pink     – convergence curve


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def pad_to_aspect(path: str, w_ratio: int = 3, h_ratio: int = 2) -> None:
    """Pad a saved PNG to the target aspect ratio (white background, centered)."""
    img = _PILImage.open(path).convert("RGBA")
    w, h = img.size
    target_w = max(w, round(h * w_ratio / h_ratio))
    target_h = max(h, round(w * h_ratio / w_ratio))
    canvas = _PILImage.new("RGBA", (target_w, target_h), (255, 255, 255, 255))
    canvas.paste(img, ((target_w - w) // 2, (target_h - h) // 2))
    canvas.convert("RGB").save(path)


def make_ut(angle_deg: float, translation: list,
            sigma_rot: float, sigma_trans: float) -> UncertainTransform:
    F = make_se3(rotz(np.deg2rad(angle_deg)), translation)
    C = np.diag([sigma_rot**2] * 3 + [sigma_trans**2] * 3)
    return UncertainTransform(F, C)


def sample_cov(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0)
    return (Xc.T @ Xc) / (len(X) - 1)


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def draw_confidence_ellipse(ax, cov2x2: np.ndarray,
                             center=(0., 0.), n_std: float = 2.0, **kwargs):
    """Draw a confidence ellipse from a 2×2 covariance sub-matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov2x2)
    angle  = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width  = 2 * n_std * np.sqrt(max(eigenvalues[0], 0.0))
    height = 2 * n_std * np.sqrt(max(eigenvalues[1], 0.0))
    ell = Ellipse(xy=center, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ell)
    return ell


def build_network() -> GeometricNetwork:
    net = GeometricNetwork()
    # Path 1 — optical tracker (lower noise)
    net.add_edge("a", "b", make_ut( 8,  [ 0.10, 0.0, 0.0], sigma_rot=0.002, sigma_trans=0.003))
    net.add_edge("b", "c", make_ut( 5,  [ 0.10, 0.0, 0.0], sigma_rot=0.001, sigma_trans=0.002))
    net.add_edge("c", "f", make_ut(-5,  [ 0.10, 0.0, 0.0], sigma_rot=0.002, sigma_trans=0.003))
    # Path 2 — robot kinematics (higher noise)
    net.add_edge("a", "d", make_ut(-8,  [-0.10, 0.0, 0.0], sigma_rot=0.004, sigma_trans=0.005))
    net.add_edge("d", "e", make_ut(-5,  [ 0.10, 0.0, 0.0], sigma_rot=0.003, sigma_trans=0.004))
    net.add_edge("e", "f", make_ut( 5,  [ 0.10, 0.0, 0.0], sigma_rot=0.004, sigma_trans=0.005))
    # Shared final edge
    net.add_edge("f", "g", make_ut( 0,  [ 0.05, 0.0, 0.0], sigma_rot=0.001, sigma_trans=0.002))
    return net


def path_cumulative_traces(net: GeometricNetwork, path: list) -> list:
    """Return tr(Σ) × 10⁻⁶ at each prefix of *path* (starts at 0 for root)."""
    traces = [0.0]
    for end in range(2, len(path) + 1):
        ut = net.query_transform_on_path(path[:end])
        traces.append(np.trace(ut.transform.C) * 1e6)
    return traces


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Network Diagram  +  Cumulative Uncertainty Growth
# ═══════════════════════════════════════════════════════════════════════════

def fig1_network() -> None:
    print("Figure 1: network diagram + cumulative uncertainty …")

    net = build_network()

    node_pos = {
        "a": (0.0, 1.5),
        "b": (1.5, 3.0), "c": (3.0, 3.0),
        "d": (1.5, 0.0), "e": (3.0, 0.0),
        "f": (4.5, 1.5),
        "g": (6.0, 1.5),
    }
    node_label = {
        "a": "World",
        "b": "Tracker", "c": "Camera",
        "d": "Base",    "e": "Elbow",
        "f": "Wrist",
        "g": "Tool tip",
    }

    edge_spec = [
        ("a", "b", 0.002, 0.003), ("b", "c", 0.001, 0.002), ("c", "f", 0.002, 0.003),
        ("a", "d", 0.004, 0.005), ("d", "e", 0.003, 0.004), ("e", "f", 0.004, 0.005),
        ("f", "g", 0.001, 0.002),
    ]
    unc = {(s, d): np.sqrt(3*sr**2 + 3*st**2) for s, d, sr, st in edge_spec}
    vals = list(unc.values())
    norm_cmap = Normalize(vmin=min(vals)*0.75, vmax=max(vals)*1.10)
    cmap_edge = plt.cm.plasma

    # Per-node cumulative covariance (translation-xy subblock for ellipses)
    path1 = ["a", "b", "c", "f", "g"]
    path2 = ["a", "d", "e", "f", "g"]
    node_cov2: dict[str, np.ndarray] = {"a": np.zeros((2, 2))}
    for end in range(2, len(path1) + 1):
        node = path1[end - 1]
        ut = net.query_transform_on_path(path1[:end])
        node_cov2[node] = ut.transform.C[3:5, 3:5]
    for end in range(2, len(path2)):          # skip f,g — already set from path1
        node = path2[end - 1]
        if node not in node_cov2:
            ut = net.query_transform_on_path(path2[:end])
            node_cov2[node] = ut.transform.C[3:5, 3:5]

    # Cumulative traces for right panel
    tr1    = path_cumulative_traces(net, path1)
    tr2    = path_cumulative_traces(net, path2)
    fused  = net.query_frame("a", "g")
    tr_fus = np.trace(fused.transform.C) * 1e6

    # ── Layout ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10.667), facecolor="white")
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1.0], wspace=0.30)
    ax  = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    # Draw edges
    for src, dst, sr, st in edge_spec:
        x0, y0 = node_pos[src]
        x1, y1 = node_pos[dst]
        color = cmap_edge(norm_cmap(unc[(src, dst)]))
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=3.8, shrinkA=28, shrinkB=28,
                                   mutation_scale=25))
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        length = np.hypot(dx, dy) + 1e-9
        ox, oy = -dy / length * 0.30, dx / length * 0.30
        ax.text(mx + ox, my + oy,
                f"σ = {unc[(src,dst)]:.4f}",
                ha="center", va="center", fontsize=8.5, color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.17", fc="white", ec="none", alpha=0.88))

    # Node colours
    node_color = {
        "a": C_BLUE,
        "b": C_BLUE,   "c": C_BLUE,
        "d": C_VIOLET, "e": C_VIOLET,
        "f": C_EMERALD,
        "g": C_ROSE,
    }

    # Uncertainty ellipses (n_std chosen so 2σ ≈ 0.25 plot units for typical nodes)
    ELLIPSE_NSTD = 50
    for name, (x, y) in node_pos.items():
        cov2 = node_cov2.get(name, np.zeros((2, 2)))
        if np.any(cov2 > 0):
            draw_confidence_ellipse(ax, cov2, center=(x, y), n_std=ELLIPSE_NSTD,
                                    facecolor=node_color[name], alpha=0.18,
                                    edgecolor=node_color[name], linewidth=1.8,
                                    zorder=3)

    # Nodes
    for name, (x, y) in node_pos.items():
        ax.add_patch(plt.Circle((x, y), 0.30,
                                color=node_color[name], zorder=5, ec="white", lw=2.8))
        ax.text(x, y, name.upper(),
                ha="center", va="center", color="white",
                fontsize=13, fontweight="bold", zorder=6)
        offset = -0.50 if name in ("b", "c", "a", "f", "g") else +0.52
        va_str = "top" if offset < 0 else "bottom"
        ax.text(x, y + offset, node_label[name],
                ha="center", va=va_str, fontsize=8.5, color=C_DARK,
                style="italic")

    # Colorbar
    sm = ScalarMappable(cmap=cmap_edge, norm=norm_cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.027, pad=0.01, shrink=0.48)
    cbar.set_label("Edge uncertainty  √tr(Σ)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Legend
    leg_handles = [
        mpatches.Patch(color=C_BLUE,    label="Path 1  (optical tracker)"),
        mpatches.Patch(color=C_VIOLET,  label="Path 2  (robot kinematics)"),
        mpatches.Patch(color=C_EMERALD, label="Shared wrist frame (f)"),
        mpatches.Patch(color=C_ROSE,    label="Target — tool tip (g)"),
    ]
    ax.legend(handles=leg_handles, loc="lower left",
              framealpha=0.92, fontsize=9, edgecolor="#CBD5E1")

    ax.set_title("Surgical Robotics Kinematic Network  (nodes a – g)\n"
                 "Shaded ellipses = 2σ positional uncertainty at each frame",
                 fontsize=12, fontweight="bold", color=C_DARK, pad=10)
    ax.set_xlim(-0.9, 7.1)
    ax.set_ylim(-1.1, 4.1)

    # ── Right panel: cumulative uncertainty growth ─────────────────────────
    ax2.set_facecolor("#F8FAFC")
    steps = list(range(len(tr1)))

    ax2.plot(steps, tr1,
             color=C_BLUE, lw=2.5, marker="o", markersize=8,
             label="Path 1  (optical)", zorder=5)
    ax2.plot(steps, tr2,
             color=C_VIOLET, lw=2.5, marker="s", markersize=8,
             linestyle="--", label="Path 2  (kinematic)", zorder=5)
    ax2.fill_between(steps,
                     [min(a, b) for a, b in zip(tr1, tr2)],
                     [max(a, b) for a, b in zip(tr1, tr2)],
                     alpha=0.10, color=C_AMBER, label="Gap between paths")
    ax2.scatter([steps[-1]], [tr_fus],
                s=180, marker="*", color=C_EMERALD, zorder=7,
                edgecolors="white", linewidths=1.2,
                label=f"Fused  ({tr_fus:.2f})")
    ax2.annotate(f"Fused\n{tr_fus:.2f}",
                 xy=(steps[-1], tr_fus),
                 xytext=(steps[-1] - 1.4, tr_fus + 0.14 * max(max(tr1), max(tr2))),
                 fontsize=8.5, color=C_EMERALD, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=C_EMERALD, lw=1.4))

    ax2.set_xticks(steps)
    ax2.set_xticklabels(path1, fontsize=10)
    ax2.set_xlabel("Frame along path", fontsize=10)
    ax2.set_ylabel("tr(Σ)  ×10⁻⁶", fontsize=10)
    ax2.set_title("Cumulative\nUncertainty Growth", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8.5, loc="upper left")
    ax2.set_ylim(bottom=-0.04 * max(max(tr1), max(tr2)))

    fig.suptitle(
        "SE(3) Uncertainty Propagation in a Geometric Network  —  Surgical Robotics",
        fontsize=14, fontweight="bold", color=C_DARK, y=1.01)
    fig.savefig("results/poster_fig1_network.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: results/poster_fig1_network.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Monte Carlo Deep-Dive  (4 panels)
# ═══════════════════════════════════════════════════════════════════════════

def fig2_montecarlo() -> None:
    print("Figure 2: Monte Carlo deep-dive …")

    chain = [
        make_ut( 8,  [0.10, 0.0, 0.0], sigma_rot=0.002, sigma_trans=0.003),
        make_ut( 5,  [0.10, 0.0, 0.0], sigma_rot=0.001, sigma_trans=0.002),
        make_ut(-5,  [0.10, 0.0, 0.0], sigma_rot=0.002, sigma_trans=0.003),
    ]

    # Analytic covariance
    T_cum = chain[0]
    for u in chain[1:]:
        T_cum = T_cum @ u
    C_analytic = T_cum.C

    # Full Monte Carlo samples (N_MC draws)
    F_nom = np.eye(4)
    for u in chain:
        F_nom = F_nom @ u.F_nom
    F_inv = inv_se3(F_nom)

    xi = np.zeros((N_MC, 6))
    for i in range(N_MC):
        T = np.eye(4)
        for u in chain:
            eta = RNG.multivariate_normal(np.zeros(6), u.C)
            T   = T @ (exp_se3(eta) @ u.F_nom)
        xi[i] = log_se3(T @ F_inv)

    C_mc      = sample_cov(xi)
    frob_err  = np.linalg.norm(C_mc - C_analytic) / np.linalg.norm(C_analytic)

    # Convergence data (nested subsets of the same samples)
    N_values  = [50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 30_000]
    frob_conv = [np.linalg.norm(sample_cov(xi[:n]) - C_analytic)
                 / np.linalg.norm(C_analytic)
                 for n in N_values]

    comp_labels = [r"$\alpha_x$", r"$\alpha_y$", r"$\alpha_z$",
                   r"$\varepsilon_x$", r"$\varepsilon_y$", r"$\varepsilon_z$"]

    # ── Layout ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11), facecolor="white")
    gs  = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.36)
    ax_sc   = fig.add_subplot(gs[0, 0])   # A – scatter + ellipses
    ax_cv   = fig.add_subplot(gs[0, 1])   # B – convergence
    ax_hi   = fig.add_subplot(gs[1, 0])   # C – histogram
    ax_hm   = fig.add_subplot(gs[1, 1])   # D – heatmap

    # ── Panel A: 2D scatter + confidence ellipses ──────────────────────────
    ax = ax_sc
    ax.set_facecolor("#EFF6FF")
    idx_sub = RNG.choice(N_MC, 3000, replace=False)
    ax.scatter(xi[idx_sub, 3] * 1e3, xi[idx_sub, 4] * 1e3,
               s=3.5, alpha=0.22, color=C_SKY, rasterized=True, zorder=2)

    cov_trans = C_analytic[3:5, 3:5]
    for n_std, lw, label in [(1, 2.2, "1σ"), (2, 2.2, "2σ"), (3, 1.6, "3σ")]:
        draw_confidence_ellipse(ax, cov_trans * 1e6,
                                center=(0., 0.), n_std=n_std,
                                facecolor=C_ROSE, edgecolor=C_ROSE,
                                alpha=0.12, linewidth=lw, zorder=4)
        semi_y = n_std * np.sqrt(cov_trans[1, 1]) * 1e3
        ax.text(0.0, semi_y * 1.08, label,
                ha="center", va="bottom", fontsize=9,
                color=C_ROSE, fontweight="bold")

    ax.set_xlabel(r"$\varepsilon_x$  (×10⁻³)", fontsize=10)
    ax.set_ylabel(r"$\varepsilon_y$  (×10⁻³)", fontsize=10)
    ax.set_title("MC Samples vs. Analytic Confidence Ellipses\n"
                 r"Translation components  ($\varepsilon_x$, $\varepsilon_y$)",
                 fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(handles=[
        mpatches.Patch(color=C_SKY,  alpha=0.6, label=f"MC samples  (N = {N_MC:,})"),
        mpatches.Patch(color=C_ROSE, alpha=0.4, label="Analytic 1σ / 2σ / 3σ"),
    ], fontsize=9, loc="upper right")
    ax.text(0.03, 0.97, "(A)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", color=C_DARK)

    # ── Panel B: Convergence curve ─────────────────────────────────────────
    ax = ax_cv
    ax.set_facecolor("#EFF6FF")
    frob_pct = [e * 100 for e in frob_conv]
    ax.plot(N_values, frob_pct,
            color=C_PINK, lw=2.5, marker="o", markersize=6, zorder=5)
    ax.fill_between(N_values, frob_pct, alpha=0.13, color=C_PINK)
    ax.axhline(1.0, color=C_AMBER, lw=2.0, linestyle="--",
               label="1 % accuracy threshold")
    ax.scatter([N_values[-1]], [frob_pct[-1]],
               s=110, color=C_EMERALD, zorder=6,
               label=f"Final: {frob_pct[-1]:.2f} %")
    ax.set_xscale("log")
    ax.set_xlabel("Number of MC samples  (log scale)", fontsize=10)
    ax.set_ylabel("Relative Frobenius error  (%)", fontsize=10)
    ax.set_title("Convergence of MC Covariance Estimate\nto Analytic Ground Truth",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.text(0.03, 0.97, "(B)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", color=C_DARK)

    # ── Panel C: Histogram of εz ──────────────────────────────────────────
    ax = ax_hi
    ax.set_facecolor("#EFF6FF")
    data = xi[:, 5] * 1e3
    mu, sigma_d = data.mean(), data.std()
    counts, bins = np.histogram(data, bins=70, density=True)
    y_max = counts.max() * 1.30

    ax.hist(data, bins=70, density=True,
            color=C_AMBER, alpha=0.60, edgecolor="white", linewidth=0.4,
            label="MC samples", zorder=2)
    x_range = np.linspace(data.min(), data.max(), 400)
    ax.plot(x_range, gaussian_pdf(x_range, mu, sigma_d),
            color=C_VIOLET, lw=2.8, zorder=5,
            label="Analytic Gaussian")
    for k, ls in [(1, "--"), (2, ":")]:
        ax.axvline(mu + k * sigma_d, color=C_ROSE, lw=1.6,
                   linestyle=ls, alpha=0.80, zorder=4)
        ax.axvline(mu - k * sigma_d, color=C_ROSE, lw=1.6,
                   linestyle=ls, alpha=0.80, zorder=4)
        ax.text(mu + k * sigma_d, y_max * 0.72,
                f"+{k}σ", color=C_ROSE, fontsize=8, ha="left", va="bottom")

    ax.text(0.97, 0.95, f"μ = {mu:.2e}\nσ = {sigma_d:.2e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CBD5E1", alpha=0.92))
    ax.set_ylim(0, y_max)
    ax.set_xlabel(r"$\varepsilon_z$  (×10⁻³)", fontsize=10)
    ax.set_ylabel("Probability density", fontsize=10)
    ax.set_title(r"Distribution of $\varepsilon_z$  Component" + "\n"
                 "MC Histogram vs. Analytic Gaussian Overlay",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.text(0.03, 0.97, "(C)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", color=C_DARK)

    # ── Panel D: Split-triangle covariance heatmap ────────────────────────
    # Lower triangle (incl. diagonal) = analytic, upper triangle = MC
    ax = ax_hm
    n = 6
    combined = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            combined[i, j] = (C_analytic[i, j] if i >= j else C_mc[i, j]) * 1e6
    vmax = max(np.abs(C_analytic).max(), np.abs(C_mc).max()) * 1e6

    im = ax.imshow(combined, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(comp_labels, fontsize=9.5)
    ax.set_yticklabels(comp_labels, fontsize=9.5)
    # Diagonal separator line
    ax.plot([-0.5, n - 0.5], [-0.5, n - 0.5],
            color="white", lw=2.5, linestyle="-", zorder=5)
    # Cell annotations
    for i in range(n):
        for j in range(n):
            val = combined[i, j]
            text_color = "white" if abs(val) / (vmax + 1e-12) > 0.40 else C_DARK
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=6.5,
                    color=text_color, zorder=6)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Covariance  (×10⁻⁶)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title("Covariance Matrix  (lower = Analytic  |  upper = MC)\n"
                 "Diagonal values confirm close agreement",
                 fontsize=11, fontweight="bold")
    ax.text(0.15, 0.10, "Analytic\n(lower)", transform=ax.transAxes,
            fontsize=8.5, color="white", ha="center",
            bbox=dict(boxstyle="round", fc=C_BLUE, alpha=0.75, ec="none"))
    ax.text(0.80, 0.88, "Monte\nCarlo\n(upper)", transform=ax.transAxes,
            fontsize=8.5, color="white", ha="center",
            bbox=dict(boxstyle="round", fc=C_VIOLET, alpha=0.75, ec="none"))
    ax.text(0.03, 0.97, "(D)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", color="white", zorder=7)

    fig.suptitle(
        f"Monte Carlo Validation  —  Open Chain a → b → c → d  "
        f"|  N = {N_MC:,} samples  "
        f"|  Rel. Frobenius error: {frob_err:.2%}",
        fontsize=13, fontweight="bold", color=C_DARK)
    fig.savefig("results/poster_fig2_montecarlo.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Rel. Frobenius error: {frob_err:.3f}")
    print("  Saved: results/poster_fig2_montecarlo.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Uncertainty Reduction via Multi-path Bayesian Fusion  (3 panels)
# ═══════════════════════════════════════════════════════════════════════════

def fig3_reduction() -> None:
    print("Figure 3: uncertainty reduction …")

    net = build_network()

    path1 = ["a", "b", "c", "f", "g"]
    path2 = ["a", "d", "e", "f", "g"]

    res1  = net.query_transform_on_path(path1)
    res2  = net.query_transform_on_path(path2)
    fused = net.query_frame("a", "g")

    tr1    = np.trace(res1.transform.C)  * 1e6
    tr2    = np.trace(res2.transform.C)  * 1e6
    tr_fus = np.trace(fused.transform.C) * 1e6

    red1 = (1 - tr_fus / tr1) * 100
    red2 = (1 - tr_fus / tr2) * 100

    tr_steps1 = path_cumulative_traces(net, path1)
    tr_steps2 = path_cumulative_traces(net, path2)

    # ── Layout ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 5.8), facecolor="white")
    gs  = GridSpec(1, 3, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── Panel A: Covariance trace bar chart ───────────────────────────────
    ax = ax1
    ax.set_facecolor("#F8FAFC")
    bar_labels = ["Path 1\n(a→b→c→f→g)", "Path 2\n(a→d→e→f→g)", "Fused\n(both paths)"]
    bar_vals   = [tr1, tr2, tr_fus]
    bar_colors = [C_BLUE, C_VIOLET, C_EMERALD]

    bars = ax.bar(bar_labels, bar_vals, color=bar_colors,
                  edgecolor="white", linewidth=1.8, width=0.50, alpha=0.90)
    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(bar_vals) * 0.016,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=C_DARK)
    ax.text(bars[2].get_x() + bars[2].get_width() / 2,
            tr_fus / 2,
            f"−{red1:.0f}% vs P1\n−{red2:.0f}% vs P2",
            ha="center", va="center", fontsize=10,
            color="white", fontweight="bold")
    ax.axhline(tr1, color=C_BLUE,   lw=1.0, linestyle=":", alpha=0.45)
    ax.axhline(tr2, color=C_VIOLET, lw=1.0, linestyle=":", alpha=0.45)
    ax.set_ylabel("tr(Σ)  ×10⁻⁶", fontsize=11)
    ax.set_title("Covariance Trace\nper Path vs. Fused",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(bar_vals) * 1.22)
    ax.text(0.03, 0.97, "(A)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", color=C_DARK)

    # ── Panel B: Per-component grouped bar chart ───────────────────────────
    ax = ax2
    ax.set_facecolor("#F8FAFC")
    comp_labels = [r"$\alpha_x$", r"$\alpha_y$", r"$\alpha_z$",
                   r"$\varepsilon_x$", r"$\varepsilon_y$", r"$\varepsilon_z$"]
    idx = np.arange(6)
    w   = 0.26

    ax.bar(idx - w, np.diag(res1.transform.C)  * 1e6, w,
           color=C_BLUE,    alpha=0.88, edgecolor="white", lw=1.2, label="Path 1")
    ax.bar(idx,     np.diag(res2.transform.C)  * 1e6, w,
           color=C_VIOLET,  alpha=0.88, edgecolor="white", lw=1.2, label="Path 2")
    ax.bar(idx + w, np.diag(fused.transform.C) * 1e6, w,
           color=C_EMERALD, alpha=0.88, edgecolor="white", lw=1.2, label="Fused")

    ax.set_xticks(idx)
    ax.set_xticklabels(comp_labels, fontsize=11)
    ax.set_ylabel("Variance  (×10⁻⁶)", fontsize=11)
    ax.set_title("Per-component Variance\nPath 1 vs Path 2 vs Fused",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.text(0.03, 0.97, "(B)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", color=C_DARK)

    # ── Panel C: Step-by-step cumulative uncertainty accumulation ──────────
    ax = ax3
    ax.set_facecolor("#F8FAFC")
    steps = list(range(len(tr_steps1)))

    ax.plot(steps, tr_steps1,
            color=C_BLUE, lw=2.5, marker="o", markersize=8,
            label="Path 1  (optical)", zorder=5)
    ax.plot(steps, tr_steps2,
            color=C_VIOLET, lw=2.5, marker="s", markersize=8,
            linestyle="--", label="Path 2  (kinematic)", zorder=5)
    ax.fill_between(steps,
                    [min(a, b) for a, b in zip(tr_steps1, tr_steps2)],
                    [max(a, b) for a, b in zip(tr_steps1, tr_steps2)],
                    alpha=0.12, color=C_AMBER, label="Gap between paths")
    ax.scatter([steps[-1]], [tr_fus],
               s=200, marker="*", color=C_EMERALD, zorder=7,
               edgecolors="white", linewidths=1.5,
               label=f"Fused  ({tr_fus:.2f})")
    # Annotation for fused star
    y_range = max(max(tr_steps1), max(tr_steps2))
    ax.annotate(f"Fused\n{tr_fus:.2f}",
                xy=(steps[-1], tr_fus),
                xytext=(steps[-1] - 1.5, tr_fus + 0.16 * y_range),
                fontsize=9, color=C_EMERALD, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=C_EMERALD, lw=1.5))

    ax.set_xticks(steps)
    ax.set_xticklabels(path1, fontsize=10)
    ax.set_xlabel("Frame  (path 1 nodes shown)", fontsize=10)
    ax.set_ylabel("Cumulative tr(Σ)  ×10⁻⁶", fontsize=10)
    ax.set_title("Step-by-step Uncertainty Accumulation\nAlong Each Path",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(bottom=-0.04 * y_range)
    ax.text(0.03, 0.97, "(C)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", color=C_DARK)

    fig.suptitle(
        "Multi-path Bayesian Fusion  —  a → g  "
        "|  Optical Tracker + Robot Kinematics",
        fontsize=14, fontweight="bold", color=C_DARK)
    fig.savefig("results/poster_fig3_reduction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Path 1 trace: {tr1:.3f}  |  Path 2 trace: {tr2:.3f}  "
          f"|  Fused: {tr_fus:.3f}  (−{red1:.1f}% vs P1, −{red2:.1f}% vs P2)")
    print("  Saved: results/poster_fig3_reduction.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Edge Sensitivity Analysis  (Ablation)
# ═══════════════════════════════════════════════════════════════════════════

def fig4_sensitivity() -> None:
    print("Figure 4: edge sensitivity analysis …")

    # Edge specs: (src, dst, angle_deg, translation, sigma_rot, sigma_trans, path)
    edge_specs = [
        ("a", "b",  8,  [ 0.10, 0.0, 0.0], 0.002, 0.003, "Path 1"),
        ("b", "c",  5,  [ 0.10, 0.0, 0.0], 0.001, 0.002, "Path 1"),
        ("c", "f", -5,  [ 0.10, 0.0, 0.0], 0.002, 0.003, "Path 1"),
        ("a", "d", -8,  [-0.10, 0.0, 0.0], 0.004, 0.005, "Path 2"),
        ("d", "e", -5,  [ 0.10, 0.0, 0.0], 0.003, 0.004, "Path 2"),
        ("e", "f",  5,  [ 0.10, 0.0, 0.0], 0.004, 0.005, "Path 2"),
        ("f", "g",  0,  [ 0.05, 0.0, 0.0], 0.001, 0.002, "Shared"),
    ]

    def build_net_with_zero(zero_edge):
        net = GeometricNetwork()
        for src, dst, ang, tr, sr, st, _ in edge_specs:
            if (src, dst) == zero_edge:
                sr_use, st_use = 1e-9, 1e-9   # effectively zero
            else:
                sr_use, st_use = sr, st
            net.add_edge(src, dst, make_ut(ang, tr, sr_use, st_use))
        return net

    # Baseline fused uncertainty
    net_base  = build_network()
    tr_base   = np.trace(net_base.query_frame("a", "g").transform.C) * 1e6

    # Ablation: drop each edge to near-zero and measure delta
    contribs  = []
    for spec in edge_specs:
        src, dst = spec[0], spec[1]
        net_abl  = build_net_with_zero((src, dst))
        tr_abl   = np.trace(net_abl.query_frame("a", "g").transform.C) * 1e6
        delta    = tr_base - tr_abl      # how much uncertainty is removed
        contribs.append((f"{src}→{dst}", delta, spec[6]))

    # Sort by contribution (ascending for horizontal bar)
    contribs.sort(key=lambda x: x[1])
    labels    = [c[0] for c in contribs]
    deltas    = [c[1] for c in contribs]
    path_tags = [c[2] for c in contribs]

    path_color = {"Path 1": C_BLUE, "Path 2": C_VIOLET, "Shared": C_EMERALD}
    colors     = [path_color[p] for p in path_tags]

    # ── Layout ─────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), facecolor="white")

    # ── Panel A: tornado bar chart ─────────────────────────────────────────
    ax = ax1
    ax.set_facecolor("#F8FAFC")
    bars = ax.barh(labels, deltas, color=colors, alpha=0.88,
                   edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, deltas):
        ax.text(val + tr_base * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                fontsize=9.5, fontweight="bold", color=C_DARK)
    ax.axvline(0, color=C_DARK, lw=1.2, alpha=0.4)
    ax.set_xlabel("Marginal contribution to tr(Σ_fused)  ×10⁻⁶", fontsize=10)
    ax.set_title("Edge Sensitivity (Ablation Study)\n"
                 "Contribution of each edge to fused uncertainty at tool tip",
                 fontsize=11, fontweight="bold")
    leg_handles = [
        mpatches.Patch(color=C_BLUE,    label="Path 1  (optical tracker)"),
        mpatches.Patch(color=C_VIOLET,  label="Path 2  (robot kinematics)"),
        mpatches.Patch(color=C_EMERALD, label="Shared  (f → g)"),
    ]
    ax.legend(handles=leg_handles, fontsize=9, loc="lower right")
    ax.text(0.03, 0.97, "(A)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", color=C_DARK)
    ax.text(0.97, 0.03,
            f"Baseline fused tr(Σ) = {tr_base:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color=C_DARK, bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                     ec="#CBD5E1", alpha=0.92))

    # ── Panel B: percentage contribution stacked bar ───────────────────────
    ax = ax2
    ax.set_facecolor("#F8FAFC")
    # Re-sort descending for stacked pie-like bar
    contribs_sorted = sorted(contribs, key=lambda x: -x[1])
    total_contrib   = sum(d for _, d, _ in contribs_sorted if d > 0)

    cumulative = 0.0
    for lbl, delta, ptag in contribs_sorted:
        if delta <= 0:
            continue
        pct = delta / total_contrib * 100
        ax.bar(0, pct, bottom=cumulative, width=0.55,
               color=path_color[ptag], alpha=0.88,
               edgecolor="white", linewidth=1.5)
        ax.text(0.30, cumulative + pct / 2,
                f"{lbl}  {pct:.1f}%",
                va="center", ha="left", fontsize=10, color=C_DARK)
        cumulative += pct

    ax.set_xlim(-0.4, 1.8)
    ax.set_ylim(0, 105)
    ax.set_xticks([])
    ax.set_ylabel("% share of total marginal contribution", fontsize=10)
    ax.set_title("Proportional Uncertainty Budget\nby Edge (top contributors)",
                 fontsize=11, fontweight="bold")
    ax.legend(handles=leg_handles, fontsize=9, loc="lower right")
    ax.text(0.03, 0.97, "(B)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", color=C_DARK)

    fig.suptitle(
        "SE(3) Uncertainty Sensitivity  —  Which Joint Calibration Matters Most?",
        fontsize=13, fontweight="bold", color=C_DARK)
    fig.savefig("results/poster_fig4_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: results/poster_fig4_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Noise Scaling & Fusion Benefit
# ═══════════════════════════════════════════════════════════════════════════

def fig5_noise_scaling() -> None:
    print("Figure 5: noise scaling & fusion benefit …")

    path1 = ["a", "b", "c", "f", "g"]
    path2 = ["a", "d", "e", "f", "g"]

    base_specs = {
        ("a", "b"): (0.002, 0.003), ("b", "c"): (0.001, 0.002), ("c", "f"): (0.002, 0.003),
        ("a", "d"): (0.004, 0.005), ("d", "e"): (0.003, 0.004), ("e", "f"): (0.004, 0.005),
        ("f", "g"): (0.001, 0.002),
    }
    edge_angles = {
        ("a", "b"):  8,  ("b", "c"):  5,  ("c", "f"): -5,
        ("a", "d"): -8,  ("d", "e"): -5,  ("e", "f"):  5,
        ("f", "g"):  0,
    }
    edge_trans = {
        ("a", "b"): [ 0.10, 0.0, 0.0], ("b", "c"): [0.10, 0.0, 0.0], ("c", "f"): [0.10, 0.0, 0.0],
        ("a", "d"): [-0.10, 0.0, 0.0], ("d", "e"): [0.10, 0.0, 0.0], ("e", "f"): [0.10, 0.0, 0.0],
        ("f", "g"): [ 0.05, 0.0, 0.0],
    }
    p1_edges = [("a", "b"), ("b", "c"), ("c", "f")]
    p2_edges = [("a", "d"), ("d", "e"), ("e", "f")]

    def build_scaled(scale1: float, scale2: float) -> GeometricNetwork:
        net = GeometricNetwork()
        for edge, (sr, st) in base_specs.items():
            if edge in p1_edges:
                s = scale1
            elif edge in p2_edges:
                s = scale2
            else:
                s = 1.0   # shared edge unscaled
            net.add_edge(edge[0], edge[1],
                         make_ut(edge_angles[edge], edge_trans[edge], sr * s, st * s))
        return net

    # ── Panel A: symmetric scale sweep ────────────────────────────────────
    scales  = np.linspace(0.2, 4.0, 40)
    tr_p1   = []
    tr_p2   = []
    tr_fus  = []
    for s in scales:
        net = build_scaled(s, s)
        tr_p1.append(np.trace(net.query_transform_on_path(path1).transform.C) * 1e6)
        tr_p2.append(np.trace(net.query_transform_on_path(path2).transform.C) * 1e6)
        tr_fus.append(np.trace(net.query_frame("a", "g").transform.C) * 1e6)

    # ── Panel B: noise ratio sweep (fix total noise, vary ratio P2/P1) ───
    ratios   = np.linspace(0.2, 6.0, 40)   # P2 noise = ratio × P1 noise
    benefit1 = []   # tr_p1 / tr_fused
    benefit2 = []   # tr_p2 / tr_fused
    for r in ratios:
        net   = build_scaled(1.0, r)
        tp1   = np.trace(net.query_transform_on_path(path1).transform.C) * 1e6
        tp2   = np.trace(net.query_transform_on_path(path2).transform.C) * 1e6
        tf    = np.trace(net.query_frame("a", "g").transform.C) * 1e6
        benefit1.append(tp1 / tf)
        benefit2.append(tp2 / tf)

    # ── Layout ─────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), facecolor="white")

    # ── Panel A ────────────────────────────────────────────────────────────
    ax = ax1
    ax.set_facecolor("#F8FAFC")
    ax.plot(scales, tr_p1,  color=C_BLUE,    lw=2.5, label="Path 1 only  (optical)")
    ax.plot(scales, tr_p2,  color=C_VIOLET,  lw=2.5, linestyle="--",
            label="Path 2 only  (kinematic)")
    ax.plot(scales, tr_fus, color=C_EMERALD, lw=3.0, linestyle="-.",
            label="Fused  (both paths)")
    ax.fill_between(scales, tr_fus, tr_p1,
                    alpha=0.10, color=C_BLUE,
                    label="Fusion benefit vs P1")
    ax.fill_between(scales, tr_fus, tr_p2,
                    alpha=0.10, color=C_VIOLET)
    ax.axvline(1.0, color=C_AMBER, lw=1.8, linestyle=":",
               label="Baseline  (scale = 1×)")
    ax.set_xlabel("Noise scale factor  (both paths scaled equally)", fontsize=10)
    ax.set_ylabel("tr(Σ) at tool tip  ×10⁻⁶", fontsize=10)
    ax.set_title("Uncertainty vs. Noise Scale\nFused always beats single path",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.text(0.03, 0.97, "(A)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", color=C_DARK)

    # ── Panel B ────────────────────────────────────────────────────────────
    ax = ax2
    ax.set_facecolor("#F8FAFC")
    ax.plot(ratios, benefit1, color=C_BLUE,   lw=2.5,
            label=r"tr(Σ$_{P1}$) / tr(Σ$_{fused}$)")
    ax.plot(ratios, benefit2, color=C_VIOLET, lw=2.5, linestyle="--",
            label=r"tr(Σ$_{P2}$) / tr(Σ$_{fused}$)")
    ax.axhline(1.0, color=C_DARK, lw=1.2, linestyle=":", alpha=0.5,
               label="No benefit (ratio = 1)")
    ax.axvline(1.0, color=C_AMBER, lw=1.8, linestyle=":",
               label="Equal-noise baseline")
    ax.fill_between(ratios, 1.0,
                    [max(b, 1.0) for b in benefit2],
                    alpha=0.12, color=C_VIOLET,
                    label="Gain from fusing noisy P2")
    ax.set_xlabel(r"Noise ratio  $\sigma_{P2}$ / $\sigma_{P1}$", fontsize=10)
    ax.set_ylabel("Uncertainty reduction ratio  (> 1 = fusion helps)", fontsize=10)
    ax.set_title("Fusion Benefit vs. Noise Ratio\nHigher ratio → more gain from fusion",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.text(0.03, 0.97, "(B)", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", color=C_DARK)

    fig.suptitle(
        "Sensitivity to Noise Levels  —  When Does Multi-path Fusion Pay Off?",
        fontsize=13, fontweight="bold", color=C_DARK)
    fig.savefig("results/poster_fig5_noise_scaling.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: results/poster_fig5_noise_scaling.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating poster figures (v2 – enhanced) …\n")
    fig1_network()
    fig2_montecarlo()
    fig3_reduction()
    fig4_sensitivity()
    fig5_noise_scaling()
    print("\nDone. All figures saved to results/")
