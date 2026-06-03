# Author: X.M. Christine Zhu
# !/usr/bin/env python3

"""
Visualise the effect of Bernstein T2R3 calibration on Galen FK tip.

Run from repository root:
    python experiment/calibration/plot_calibration_effect.py

Outputs:
    experiment/calibration/fig1_correction_heatmaps.png
    experiment/calibration/fig2_tip_paths_3d.png
    experiment/calibration/fig3_tip_paths_1d.png
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
sys.path.insert(0, os.path.join(_ROOT, 'simulation'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers 3d projection)
from scipy.spatial.transform import Rotation

from node_registry import galen_fk


# ── Nominal FK (no calibration) ───────────────────────────────────────────────

def _nominal_tip(c1, c2, c3, roll, tilt):
    R_c = 0.255; h0 = 0.837
    c_avg = (c1 + c2 + c3) / 3.0
    z_mp = h0 + c_avg
    plt_pitch = (c1 - 0.5*c2 - 0.5*c3) / R_c
    plt_roll  = (c3 - c2) * (3**0.5 / 2) / R_c
    cp, sp   = np.cos(plt_pitch), np.sin(plt_pitch)
    crp, srp = np.cos(plt_roll),  np.sin(plt_roll)
    R_mp = np.array([[cp,  sp*srp, sp*crp],
                     [0,      crp,   -srp],
                     [-sp, cp*srp, cp*crp]])
    T_mp = np.eye(4); T_mp[:3, :3] = R_mp; T_mp[:3, 3] = [0, 0, z_mp]

    cr, sr = np.cos(roll), np.sin(roll)
    R_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    T_rab = np.eye(4); T_rab[:3, :3] = R_mp @ R_roll
    T_rab[:3, 3] = T_mp[:3, :3] @ np.array([0.031, 0.0, 0.058]) + T_mp[:3, 3]

    tilt_eff = tilt - 0.126
    ct, st = np.cos(tilt_eff), np.sin(tilt_eff)
    R_tj = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    T_tilt = np.eye(4); T_tilt[:3, :3] = T_rab[:3, :3] @ R_tj
    T_tilt[:3, 3] = T_rab[:3, :3] @ np.array([0.0, 0.0, 0.588]) + T_rab[:3, 3]

    T_tip = np.eye(4); T_tip[:3, :3] = T_tilt[:3, :3]
    T_tip[:3, 3] = T_tilt[:3, :3] @ np.array([0.0, 0.0, 0.032]) + T_tilt[:3, 3]
    return T_tip


def _rot_angle_deg(R1, R2):
    dR = R1.T @ R2
    cos_t = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))


# ── Compute correction field over (roll, tilt) ────────────────────────────────

print("Computing correction field (40×40 grid) …")
N = 40
rolls = np.linspace(-1.0, 1.0, N)   # rad  (~±57°)
tilts = np.linspace(-0.9, 0.9, N)   # rad  (~±52°)
RR, TT = np.meshgrid(rolls, tilts)

trans_mm = np.zeros((N, N))
rot_deg  = np.zeros((N, N))

for i, tilt in enumerate(tilts):
    for j, roll in enumerate(rolls):
        T_nom = _nominal_tip(0, 0, 0, roll, tilt)
        T_cal = galen_fk([0, 0, 0, roll, tilt])[-1]
        trans_mm[i, j] = np.linalg.norm(T_cal[:3, 3] - T_nom[:3, 3]) * 1e3
        rot_deg[i, j]  = _rot_angle_deg(T_nom[:3, :3], T_cal[:3, :3])

print(f"  Translation correction:  max={trans_mm.max():.2f} mm, "
      f"mean={trans_mm.mean():.2f} mm")
print(f"  Rotation correction:     max={rot_deg.max():.2f}°,  "
      f"mean={rot_deg.mean():.2f}°")


# ── Figure 1: Heatmaps ────────────────────────────────────────────────────────

fig1, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig1.suptitle(
    "Galen T2R3 Calibration Correction Field\n"
    "(c₁=c₂=c₃=0, calibration data from optical-tracker experiment)",
    fontsize=13, fontweight='bold')

RR_deg = np.degrees(RR)
TT_deg = np.degrees(TT)

im1 = axes[0].contourf(RR_deg, TT_deg, trans_mm, levels=25, cmap='viridis')
axes[0].contour(RR_deg, TT_deg, trans_mm, levels=12,
                colors='white', alpha=0.25, linewidths=0.5)
cb1 = plt.colorbar(im1, ax=axes[0])
cb1.set_label('Correction magnitude (mm)', fontsize=11)
axes[0].set_xlabel('Roll (°)', fontsize=11)
axes[0].set_ylabel('Tilt (°)', fontsize=11)
axes[0].set_title(
    f'Translation (T-stage)\nmax {trans_mm.max():.2f} mm  |  mean {trans_mm.mean():.2f} mm',
    fontsize=11)

im2 = axes[1].contourf(RR_deg, TT_deg, rot_deg, levels=25, cmap='plasma')
axes[1].contour(RR_deg, TT_deg, rot_deg, levels=12,
                colors='white', alpha=0.25, linewidths=0.5)
cb2 = plt.colorbar(im2, ax=axes[1])
cb2.set_label('Correction magnitude (°)', fontsize=11)
axes[1].set_xlabel('Roll (°)', fontsize=11)
axes[1].set_ylabel('Tilt (°)', fontsize=11)
axes[1].set_title(
    f'Rotation (R-stage)\nmax {rot_deg.max():.2f}°  |  mean {rot_deg.mean():.2f}°',
    fontsize=11)

plt.tight_layout()
out1 = os.path.join(_HERE, 'fig1_correction_heatmaps.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
print(f"Saved: {out1}")


# ── Figure 2: 3-D tip paths ───────────────────────────────────────────────────

fig2 = plt.figure(figsize=(14, 6))
fig2.suptitle("Galen Tip Path: Nominal vs Calibrated FK (mm)", fontsize=13,
              fontweight='bold')

SWEEPS = [
    ("Roll sweep  (c₁=c₂=c₃=0, tilt=0)", 3, 0.0),
    ("Tilt sweep  (c₁=c₂=c₃=0, roll=0)", 4, 0.0),
]

for idx, (title, vary, fixed_other) in enumerate(SWEEPS):
    ax = fig2.add_subplot(1, 2, idx + 1, projection='3d')
    vals = np.linspace(-0.9, 0.9, 80)
    p_nom = np.zeros((len(vals), 3))
    p_cal = np.zeros((len(vals), 3))
    for k, v in enumerate(vals):
        jts = [0, 0, 0, 0, 0]
        jts[vary] = v
        p_nom[k] = _nominal_tip(*jts)[:3, 3]
        p_cal[k] = galen_fk(jts)[-1][:3, 3]
    p_nom_mm = p_nom * 1e3
    p_cal_mm = p_cal * 1e3

    ax.plot(*p_nom_mm.T, '--', color='steelblue', lw=2, label='Nominal FK', alpha=0.8)
    ax.plot(*p_cal_mm.T, '-',  color='firebrick', lw=2, label='Calibrated FK')

    # Correction arrows at every 10th point
    step = max(1, len(vals) // 8)
    for k in range(0, len(vals), step):
        dv = p_cal_mm[k] - p_nom_mm[k]
        if np.linalg.norm(dv) > 0.01:
            ax.quiver(*p_nom_mm[k], *dv,
                      color='green', alpha=0.6, arrow_length_ratio=0.35, lw=1.2)

    ax.set_xlabel('X (mm)', fontsize=9)
    ax.set_ylabel('Y (mm)', fontsize=9)
    ax.set_zlabel('Z (mm)', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)

plt.tight_layout()
out2 = os.path.join(_HERE, 'fig2_tip_paths_3d.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f"Saved: {out2}")


# ── Figure 3: Per-axis correction vs joint angle ──────────────────────────────

fig3, axes3 = plt.subplots(2, 3, figsize=(14, 8), sharex='row')
fig3.suptitle(
    "Per-axis FK Correction: Nominal vs Calibrated\n"
    "(green = calibration shift; positive correction = calibrated > nominal)",
    fontsize=12, fontweight='bold')

axes3_labels = ['X (mm)', 'Y (mm)', 'Z (mm)']
sweep_configs = [
    ("Roll (°)",  3, np.linspace(-57.3, 57.3, 80),
     np.linspace(-1.0, 1.0, 80)),
    ("Tilt (°)",  4, np.linspace(-51.6, 51.6, 80),
     np.linspace(-0.9, 0.9, 80)),
]

for row, (xlabel, vary, deg_vals, rad_vals) in enumerate(sweep_configs):
    p_nom = np.zeros((len(rad_vals), 3))
    p_cal = np.zeros((len(rad_vals), 3))
    for k, v in enumerate(rad_vals):
        jts = [0, 0, 0, 0, 0]; jts[vary] = v
        p_nom[k] = _nominal_tip(*jts)[:3, 3]
        p_cal[k] = galen_fk(jts)[-1][:3, 3]
    p_nom_mm = p_nom * 1e3
    p_cal_mm = p_cal * 1e3
    diff_mm  = p_cal_mm - p_nom_mm

    for col in range(3):
        ax = axes3[row, col]
        ax.plot(deg_vals, p_nom_mm[:, col], '--', color='steelblue',
                lw=1.8, label='Nominal', alpha=0.85)
        ax.plot(deg_vals, p_cal_mm[:, col], '-',  color='firebrick',
                lw=1.8, label='Calibrated')
        ax.fill_between(deg_vals, p_nom_mm[:, col], p_cal_mm[:, col],
                        color='green', alpha=0.18, label='Correction')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(axes3_labels[col], fontsize=10)
        ax.grid(True, alpha=0.3)
        if row == 0 and col == 0:
            ax.legend(fontsize=9)
        ax.set_title(f"{axes3_labels[col].split()[0]} — {xlabel.split()[0]} sweep",
                     fontsize=10)

plt.tight_layout()
out3 = os.path.join(_HERE, 'fig3_tip_paths_1d.png')
plt.savefig(out3, dpi=150, bbox_inches='tight')
print(f"Saved: {out3}")


# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  CALIBRATION CORRECTION SUMMARY  (roll/tilt workspace)")
print("=" * 60)
print(f"  Translation correction (T-stage):")
print(f"    max  = {trans_mm.max():.2f} mm")
print(f"    mean = {trans_mm.mean():.2f} mm")
print(f"    std  = {trans_mm.std():.2f} mm")
print(f"  Rotation correction (R-stage + T-stage combined):")
print(f"    max  = {rot_deg.max():.2f}°")
print(f"    mean = {rot_deg.mean():.2f}°")
print(f"    std  = {rot_deg.std():.2f}°")
print(f"  Calibration data source:")
print(f"    bpoly_t.csv  (p=3, q=6, d=2)  → 27×6 coefficients")
print(f"    bpoly_r.csv  (p=2, q=6, d=3)  → 16×6 coefficients")
print("=" * 60)

plt.show()
print("Done.")

#The 78 mm translation correction is much larger than the calibration report's claimed 4–8 mm pre-calibration error. This likely means the bpoly_t inputs have a coordinate frame mismatch — the student's calibration computed delta_pos in their FK frame (where the arm extends in X direction), but our simulation computes it in a different frame (arm extends in Z direction). When the inputs are outside the polynomial's training range, the outputs extrapolate to large values.
#What is reliable: The bpoly_r rotation correction is frame-independent (inputs are just (roll, tilt) radians, directly from joints) and should be correct. The rotation heatmap in Fig 1 right and the rotation corrections in Fig 3 are trustworthy.
# "calibration infrastructure integrated and evaluated," but the translation correction (bpoly_t) needs coordinate frame verification. The rotation correction (bpoly_r) is the dominant and more reliable part.