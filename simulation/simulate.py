"""
Standalone animation script launched by gui.py as a subprocess.
Reads selections from a JSON file passed as argv[1].

Supports two modes (set via "mode" key in selections JSON):
  "mock" (default) — joint angles driven by sine waves, no AMBF needed.
  "live"           — joint angles streamed from ambf_bridge.py running in WSL.
"""

import sys
import os
import json
import subprocess
import threading
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'src'))

from node_registry import NODES
from uncertainty_system import build_network, mock_joint_trajectory

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

BG        = "#0d0d1a"
PANEL_BG  = "#16213e"
ACCENT    = "#0f3460"
TEXT      = "#e0e0e0"
_TRIAD    = ["#ff4444", "#44ff88", "#4488ff"]


# ── Uncertainty ellipsoid geometry ────────────────────────────────────────────

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


# ── AMBF live-state reader ─────────────────────────────────────────────────────

class AmbfBridge:
    """
    Spawns ambf_bridge.py inside WSL as a subprocess and continuously
    reads the latest joint-state JSON line in a background thread.
    Thread-safe: _latest is written only by the reader thread,
    read only by the main (animation) thread between frames.
    """

    def __init__(self, robot_names, bridge_cmd):
        self._latest = {}          # {robot_name: [joint_angles]}
        self._lock   = threading.Lock()
        self._alive  = True

        # Build the WSL command that sources ROS and runs the bridge.
        # bridge_cmd comes from the GUI (user-configurable).
        bridge_path = os.path.join(_HERE, "ambf_bridge.py").replace("\\", "/")
        # Convert Windows path to WSL mount path
        # e.g. C:\Users\... → /mnt/c/Users/...
        if bridge_path[1] == ":":
            drive = bridge_path[0].lower()
            bridge_path = f"/mnt/{drive}" + bridge_path[2:].replace("\\", "/")

        names_arg = " ".join(robot_names)
        inner_cmd = f"{bridge_cmd} && python3 {bridge_path} {names_arg}"

        self._proc = subprocess.Popen(
            ["wsl", "bash", "-lc", inner_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        for line in self._proc.stdout:
            if not self._alive:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                with self._lock:
                    self._latest = data
            except json.JSONDecodeError:
                pass

    def get(self):
        with self._lock:
            return dict(self._latest)

    def close(self):
        self._alive = False
        try:
            self._proc.terminate()
        except Exception:
            pass


# ── Main animation ─────────────────────────────────────────────────────────────

def run(selections):
    # Determine mode — every selection carries the same "mode" value.
    mode        = selections[0].get("mode", "mock") if selections else "mock"
    bridge_cmd  = selections[0].get("bridge_cmd", "") if selections else ""
    robot_names = [s["name"] for s in selections]

    bridge = None
    mode_label = "Mock"
    if mode == "live":
        try:
            bridge = AmbfBridge(robot_names, bridge_cmd)
            mode_label = "Live (AMBF)"
        except Exception as e:
            sys.stderr.write(f"[simulate] WARNING: could not start AMBF bridge: {e}\n"
                             "[simulate] Falling back to mock mode.\n")

    fig = plt.figure(figsize=(11, 8), facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d")
    fig.suptitle(
        f"Surgical Robotics — Uncertainty Propagation  [{mode_label}]",
        color=TEXT, fontsize=13, y=0.97,
    )

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

        # Resolve joint angles: live from AMBF or mock sine waves
        live_states = bridge.get() if bridge else {}

        live = []
        for sel in selections:
            s = sel.copy()
            name = sel["name"]
            if name in live_states and live_states[name]:
                s["joint_angles"] = np.array(live_states[name])
            else:
                s["joint_angles"] = mock_joint_trajectory(t_val[0], name)
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

                # uncertainty ellipsoid at tip
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

        # legend + mode indicator
        patch_list = []
        for sel in live:
            n = sel["name"]
            if n in tip_nodes:
                patch_list.append(mpatches.Patch(color=NODES[n]["color"], label=n))
        if patch_list:
            ax.legend(handles=patch_list, loc="upper left",
                      facecolor=PANEL_BG, labelcolor=TEXT, fontsize=9)

        # show AMBF connectivity status in corner
        if bridge:
            states_received = bool(bridge.get())
            status_text = "AMBF: connected" if states_received else "AMBF: waiting…"
            status_color = "#44ff88" if states_received else "#ffaa44"
            ax.text2D(0.98, 0.02, status_text, transform=ax.transAxes,
                      color=status_color, fontsize=8, ha="right")

    ani = animation.FuncAnimation(fig, update, interval=60,
                                  cache_frame_data=False)
    plt.tight_layout()
    try:
        plt.show()
    finally:
        if bridge:
            bridge.close()


if __name__ == "__main__":
    path = sys.argv[1]
    with open(path) as f:
        selections = json.load(f)
    run(selections)
