"""
Standalone animation script launched by gui.py as a subprocess.
Reads selections from a JSON file passed as argv[1].
"""

import sys
import os
import json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'src'))

from node_registry import NODES
from uncertainty_system import build_network, mock_joint_trajectory

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BG        = "#0d0d1a"
PANEL_BG  = "#16213e"
ACCENT    = "#0f3460"
TEXT      = "#e0e0e0"
_TRIAD    = ["#ff4444", "#44ff88", "#4488ff"]


def _ellipsoid(center, cov3, n_sigma=50, n_pts=20):
    eigvals, eigvecs = np.linalg.eigh(cov3)
    eigvals = np.maximum(eigvals, 1e-30)
    radii   = n_sigma * np.sqrt(eigvals)
    u = np.linspace(0, 2 * np.pi, n_pts)
    v = np.linspace(0, np.pi,     n_pts)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    ell = pts @ (eigvecs * radii).T + center
    return (ell[:, 0].reshape(n_pts, n_pts),
            ell[:, 1].reshape(n_pts, n_pts),
            ell[:, 2].reshape(n_pts, n_pts))


def run(selections):
    fig = plt.figure(figsize=(11, 8), facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d")
    fig.suptitle("Surgical Robotics — Live Uncertainty Propagation",
                 color=TEXT, fontsize=13, y=0.97)

    t_val = [0.0]

    def update(_):
        ax.cla()
        ax.set_facecolor(BG)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_xlabel("X (m)", color=TEXT, labelpad=6)
        ax.set_ylabel("Y (m)", color=TEXT, labelpad=6)
        ax.set_zlabel("Z (m)", color=TEXT, labelpad=6)
        ax.tick_params(colors=TEXT, labelsize=7)

        t_val[0] += 0.05

        live = []
        for sel in selections:
            s = sel.copy()
            s["joint_angles"] = mock_joint_trajectory(t_val[0], sel["name"])
            live.append(s)

        net, tip_nodes = build_network(live)
        all_pos = [np.zeros(3)]

        for sel in live:
            name  = sel["name"]
            node  = NODES[name]
            color = node["color"]

            if name not in tip_nodes:
                continue

            chain = (["World", f"{name}_Base"] +
                     [f"{name}_{lbl}" for lbl in node["link_labels"]])

            frames_data = []
            for frame in chain:
                try:
                    res = net.query_frame("World", frame)
                    pos = res.transform.F_nom[:3, 3]
                    rot = res.transform.F_nom[:3, :3]
                    C   = res.transform.C[3:6, 3:6]
                    frames_data.append((frame, pos, rot, C))
                    all_pos.append(pos)
                except Exception:
                    pass

            # kinematic chain edges
            for i in range(len(frames_data) - 1):
                p0, p1 = frames_data[i][1], frames_data[i + 1][1]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                        color=color, lw=2.0, alpha=0.75)

            triad_scale = 0.03
            for frame, pos, rot, cov3 in frames_data:
                is_tip = (frame == tip_nodes[name])

                ax.scatter(*pos, color=color, s=60 if is_tip else 20,
                           zorder=5, depthshade=False)

                label = frame.split("_", 1)[-1]
                ax.text(pos[0], pos[1], pos[2] + triad_scale * 0.7,
                        label, fontsize=7, color=color, ha="center")

                for i, tc in enumerate(_TRIAD):
                    d = rot[:, i] * triad_scale
                    ax.quiver(*pos, *d, color=tc, linewidth=1.0,
                              arrow_length_ratio=0.3, normalize=False)

                # ellipsoid at tip
                if is_tip and not np.allclose(cov3, 0) and np.all(np.isfinite(cov3)):
                    try:
                        X, Y, Z = _ellipsoid(pos, cov3)
                        ax.plot_surface(X, Y, Z, alpha=0.35, color=color,
                                        linewidth=0, antialiased=True)
                    except Exception:
                        pass

        # world origin
        ax.scatter(0, 0, 0, color="white", s=80, marker="*", zorder=10)
        ax.text(0, 0, 0.04, "World", fontsize=8, color="white", ha="center")

        # axis limits
        all_pos_arr = np.array(all_pos)
        center = all_pos_arr.mean(axis=0)
        half   = max(float(np.max(np.abs(all_pos_arr - center))) + 0.1, 0.2)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)

        # legend
        handles = []
        for sel in live:
            n = sel["name"]
            if n in tip_nodes:
                handles.append(mpatches.Patch(
                    color=NODES[n]["color"], label=n))
        if handles:
            ax.legend(handles=handles, loc="upper left",
                      facecolor=PANEL_BG, labelcolor=TEXT, fontsize=9)

    ani = animation.FuncAnimation(fig, update, interval=60,
                                  cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1]
    with open(path) as f:
        selections = json.load(f)
    run(selections)
