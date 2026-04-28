"""
Surgical Robotics Uncertainty Visualizer — GUI

Tkinter-based node selector. User checks robots, adjusts sigma sliders,
then clicks "Run Simulation" to open a live matplotlib animation showing
the kinematic chain with uncertainty ellipsoids updating in real time.
"""

import sys
import os
import json
import tempfile
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'src'))

_HERE = os.path.dirname(os.path.abspath(__file__))

from node_registry import NODES


# ── colours ───────────────────────────────────────────────────────────────────
BG        = "#1a1a2e"
PANEL_BG  = "#16213e"
ACCENT    = "#0f3460"
TEXT      = "#e0e0e0"
HIGHLIGHT = "#e94560"
BTN_BG    = "#0f3460"
BTN_FG    = "#ffffff"




# ── GUI ───────────────────────────────────────────────────────────────────────

class UncertaintyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Surgical Robotics Uncertainty Visualizer")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self._node_vars   = {}   # name -> BooleanVar (checkbox)
        self._sigma_joint = {}   # name -> DoubleVar  (slider)
        self._sigma_base  = {}   # name -> DoubleVar  (slider)
        self._base_pos    = {}   # name -> [DoubleVar x3]

        self._build_header()
        self._build_node_panel()
        self._build_config_panel()
        self._build_footer()

    # ── header ────────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=HIGHLIGHT, pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Surgical Robotics  —  Uncertainty Propagation Visualizer",
                 bg=HIGHLIGHT, fg="white",
                 font=("Helvetica", 14, "bold")).pack()
        tk.Label(hdr, text="Select robots, set noise levels, then Run Simulation",
                 bg=HIGHLIGHT, fg="#ffdddd",
                 font=("Helvetica", 10)).pack()

    # ── left panel: node checkboxes ───────────────────────────────────────────

    def _build_node_panel(self):
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(side="left", fill="y", padx=10, pady=10)

        tk.Label(outer, text="NODES", bg=BG, fg=HIGHLIGHT,
                 font=("Helvetica", 11, "bold")).pack(anchor="w")
        tk.Label(outer, text="Select robots to include:",
                 bg=BG, fg=TEXT, font=("Helvetica", 9)).pack(anchor="w", pady=(0, 6))

        for name, node in NODES.items():
            var = tk.BooleanVar(value=(name in ("PSM", "ECM")))
            self._node_vars[name] = var

            row = tk.Frame(outer, bg=PANEL_BG, bd=1, relief="flat",
                           pady=4, padx=6)
            row.pack(fill="x", pady=2)

            cb = tk.Checkbutton(row, text=node["label"],
                                variable=var, bg=PANEL_BG, fg=TEXT,
                                selectcolor=ACCENT, activebackground=PANEL_BG,
                                activeforeground=TEXT,
                                font=("Helvetica", 10),
                                command=self._refresh_config)
            cb.pack(anchor="w")

            dot = tk.Label(row, text="●", bg=PANEL_BG, fg=node["color"],
                           font=("Helvetica", 14))
            dot.place(relx=1.0, rely=0.5, anchor="e", x=-4)

    # ── right panel: per-robot configuration ──────────────────────────────────

    def _build_config_panel(self):
        self._config_outer = tk.Frame(self.root, bg=BG)
        self._config_outer.pack(side="left", fill="both",
                                expand=True, padx=10, pady=10)

        tk.Label(self._config_outer, text="CONFIGURATION",
                 bg=BG, fg=HIGHLIGHT,
                 font=("Helvetica", 11, "bold")).pack(anchor="w")

        self._config_canvas = tk.Canvas(self._config_outer, bg=BG,
                                        highlightthickness=0, width=460)
        self._config_canvas.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(self._config_outer, orient="vertical",
                           command=self._config_canvas.yview)
        sb.pack(side="right", fill="y")
        self._config_canvas.configure(yscrollcommand=sb.set)

        self._config_frame = tk.Frame(self._config_canvas, bg=BG)
        self._config_canvas.create_window((0, 0), window=self._config_frame,
                                          anchor="nw")
        self._config_frame.bind("<Configure>",
            lambda e: self._config_canvas.configure(
                scrollregion=self._config_canvas.bbox("all")))

        self._refresh_config()

    def _refresh_config(self):
        for w in self._config_frame.winfo_children():
            w.destroy()

        any_selected = False
        for name, var in self._node_vars.items():
            if not var.get():
                continue
            any_selected = True
            self._build_robot_config(self._config_frame, name)

        if not any_selected:
            tk.Label(self._config_frame,
                     text="(No robots selected)",
                     bg=BG, fg="#888888",
                     font=("Helvetica", 10, "italic")).pack(pady=20)

    def _build_robot_config(self, parent, name):
        node  = NODES[name]
        color = node["color"]

        card = tk.Frame(parent, bg=PANEL_BG, bd=1, relief="flat",
                        padx=10, pady=8)
        card.pack(fill="x", pady=4)

        # title
        tk.Label(card, text=f"  {name}", bg=PANEL_BG, fg=color,
                 font=("Helvetica", 11, "bold")).grid(
                     row=0, column=0, columnspan=4, sticky="w")

        # ensure vars exist
        if name not in self._sigma_joint:
            self._sigma_joint[name] = tk.DoubleVar(
                value=node["default_sigma_joint"])
            self._sigma_base[name]  = tk.DoubleVar(
                value=node["default_sigma_base"])
            self._base_pos[name]    = [
                tk.DoubleVar(value=v)
                for v in node["default_base_pos"]
            ]

        def _slider_row(parent, row, label, var, from_, to, fmt):
            tk.Label(parent, text=label, bg=PANEL_BG, fg=TEXT,
                     font=("Helvetica", 9), width=18,
                     anchor="w").grid(row=row, column=0, sticky="w", pady=2)
            s = tk.Scale(parent, variable=var, from_=from_, to=to,
                         orient="horizontal", resolution=(to - from_) / 1000,
                         bg=PANEL_BG, fg=TEXT, troughcolor=ACCENT,
                         highlightbackground=PANEL_BG,
                         activebackground=HIGHLIGHT,
                         length=180, showvalue=False)
            s.grid(row=row, column=1, sticky="ew", padx=4)
            lbl = tk.Label(parent, bg=PANEL_BG, fg=color,
                           font=("Courier", 9), width=10)
            lbl.grid(row=row, column=2, sticky="w")

            def _update_lbl(*_, _lbl=lbl, _var=var, _fmt=fmt):
                try:
                    _lbl.config(text=_fmt.format(_var.get()))
                except tk.TclError:
                    pass
            var.trace_add("write", _update_lbl)
            _update_lbl()

        _slider_row(card, 1, "Joint noise σ (rad/m)",
                    self._sigma_joint[name], 0.0001, 0.02, "{:.4f} rad")
        _slider_row(card, 2, "Base reg. σ (m)",
                    self._sigma_base[name],  0.0001, 0.02, "{:.4f} m")

        # base position
        tk.Label(card, text="Base position (x, y, z):",
                 bg=PANEL_BG, fg=TEXT,
                 font=("Helvetica", 9)).grid(
                     row=3, column=0, sticky="w", pady=(6, 2))

        pos_row = tk.Frame(card, bg=PANEL_BG)
        pos_row.grid(row=4, column=0, columnspan=4, sticky="w")
        for i, axis in enumerate(["x", "y", "z"]):
            tk.Label(pos_row, text=f"{axis}:", bg=PANEL_BG, fg=TEXT,
                     font=("Helvetica", 9)).grid(row=0, column=i * 2, padx=(4, 0))
            e = tk.Entry(pos_row, textvariable=self._base_pos[name][i],
                         width=6, bg=ACCENT, fg=TEXT,
                         insertbackground=TEXT, relief="flat",
                         font=("Courier", 10))
            e.grid(row=0, column=i * 2 + 1, padx=(2, 6))

        # Raven2 warning
        if name == "Raven2":
            tk.Label(card,
                     text="⚠  DH params not yet available — "
                          "ask mentor for Raven2 kinematics",
                     bg=PANEL_BG, fg="#ffaa44",
                     font=("Helvetica", 8, "italic"),
                     wraplength=420).grid(
                         row=5, column=0, columnspan=4,
                         sticky="w", pady=(6, 0))

    # ── footer: run button ────────────────────────────────────────────────────

    def _build_footer(self):
        footer = tk.Frame(self.root, bg=ACCENT, pady=10)
        footer.pack(fill="x", side="bottom")

        tk.Button(footer,
                  text="▶  Run Simulation & Visualize",
                  command=self._run,
                  bg=HIGHLIGHT, fg="white",
                  font=("Helvetica", 12, "bold"),
                  relief="flat", padx=20, pady=6,
                  activebackground="#c73652",
                  cursor="hand2").pack()

        self._status = tk.Label(footer, text="Ready.",
                                bg=ACCENT, fg=TEXT,
                                font=("Helvetica", 9))
        self._status.pack(pady=(4, 0))

    # ── subprocess check ──────────────────────────────────────────────────────

    def _check_proc(self, proc, log_path):
        ret = proc.poll()
        if ret is not None and ret != 0:
            try:
                with open(log_path) as f:
                    err = f.read().strip()
            except Exception:
                err = "(could not read log)"
            messagebox.showerror("Simulation Error",
                                 f"simulate.py crashed (exit {ret}):\n\n{err[-800:]}")
            self._status.config(text="Error — see popup for details.")

    # ── run ───────────────────────────────────────────────────────────────────

    def _run(self):
        selections = []
        for name, var in self._node_vars.items():
            if not var.get():
                continue
            if name == "Raven2":
                messagebox.showwarning(
                    "Raven2 not available",
                    "Raven2 DH parameters are not yet implemented.\n"
                    "It will be skipped in the simulation."
                )
                continue
            try:
                base_pos = [v.get() for v in self._base_pos[name]]
            except Exception:
                base_pos = NODES[name]["default_base_pos"]

            selections.append({
                "name":         name,
                "joint_angles": [0.0] * NODES[name]["n_joints"],
                "sigma_joint":  self._sigma_joint[name].get(),
                "sigma_base":   self._sigma_base[name].get(),
                "base_pos":     base_pos,
            })

        if not selections:
            messagebox.showinfo("Nothing to show",
                                "Please select at least one robot.")
            return

        self._status.config(text="Simulation running…  "
                                 "(close the plot window to stop)")
        self.root.update()

        # write selections to a temp file and launch simulate.py as subprocess
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False)
        json.dump(selections, tmp)
        tmp.close()

        script = os.path.join(_HERE, 'simulate.py')
        log_path = os.path.join(_HERE, 'simulate_error.log')
        log = open(log_path, 'w')
        proc = subprocess.Popen(
            [sys.executable, script, tmp.name],
            cwd=_HERE,
            stdout=log, stderr=log
        )
        # poll after 2 seconds to catch immediate crashes
        self.root.after(2000, lambda: self._check_proc(proc, log_path))


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Vertical.TScrollbar",
                    background=ACCENT, troughcolor=BG,
                    arrowcolor=TEXT)

    UncertaintyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
