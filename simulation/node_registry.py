"""
Node registry for surgical robotics uncertainty GUI.

Each entry defines a robot's kinematic chain (DH parameters),
tip link, default sigma values, and AMBF namespace for future
live connection.

DH sources:
  PSM / ECM / MTM  — ambf/core/ambf_controller/dvrk/scripts/
  Raven2           — placeholder, needs accurate DH from mentor
"""

import os
from math import comb
from itertools import product as _iprod

import numpy as np

PI   = np.pi
PI_2 = np.pi / 2


# ── Bernstein-polynomial calibration helpers ──────────────────────────────────
# Geometric constants from experiment/calibration/calibration.py (metres)
_D2R = np.array([0.0,       0.0,  76.14070118e-3])  # delta-to-roll
_R2T = np.array([608.5e-3,  0.0,  13e-3])           # roll-to-tilt
_T2F = np.array([31.248e-3, 0.0,  25e-3])           # tilt-to-finger


def _bern(x, i, d):
    return comb(d, i) * (x ** i) * ((1.0 - x) ** (d - i))


def _load_bpoly(path):
    with open(path) as f:
        lines = [[float(v) for v in ln.strip().split(',')]
                 for ln in f if ln.strip()]
    p, q, d = int(lines[0][0]), int(lines[0][1]), int(lines[0][2])
    return {'p': p, 'q': q, 'd': d,
            'min': np.array(lines[1]),
            'max': np.array(lines[2]),
            'C':   np.array(lines[3:])}


def _bpoly_eval(inputs, bp):
    p, d = bp['p'], bp['d']
    x = np.clip((inputs - bp['min']) / (bp['max'] - bp['min'] + 1e-30), 0.0, 1.0)
    bm = np.array([
        np.prod([_bern(x[j], idx[j], d) for j in range(p)])
        for idx in _iprod(range(d + 1), repeat=p)
    ])
    return bm @ bp['C']


def _v6_htm(v):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, 3] = v[:3]
    T[:3, :3] = Rotation.from_euler('XYZ', v[3:]).as_matrix()
    return T


_BPOLY_CACHE = {}


def _get_bpoly():
    if not _BPOLY_CACHE:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _coef = os.path.join(_root, "experiment", "calibration", "coef")
        try:
            _BPOLY_CACHE['t'] = _load_bpoly(os.path.join(_coef, "bpoly_t.csv"))
            _BPOLY_CACHE['r'] = _load_bpoly(os.path.join(_coef, "bpoly_r.csv"))
        except (FileNotFoundError, OSError):
            _BPOLY_CACHE['t'] = None
            _BPOLY_CACHE['r'] = None
    return _BPOLY_CACHE.get('t'), _BPOLY_CACHE.get('r')


# ── DH transform (supports Standard and Modified conventions) ─────────────────

def _dh(alpha, a, theta, d, offset, joint_type, convention):
    if joint_type == 'R':
        theta = theta + offset
    else:
        d = d + offset

    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    if convention == 'STANDARD':
        return np.array([
            [ct,  -st * ca,  st * sa,  a * ct],
            [st,   ct * ca, -ct * sa,  a * st],
            [0,         sa,       ca,       d],
            [0,          0,        0,       1],
        ], dtype=float)
    else:  # MODIFIED
        return np.array([
            [ct,      -st,    0,       a],
            [st * ca,  ct * ca, -sa, -d * sa],
            [st * sa,  ct * sa,  ca,  d * ca],
            [0,          0,       0,        1],
        ], dtype=float)


def _chain(link_specs):
    """Return list of cumulative transforms T_k_0 for k = 1..n."""
    transforms = []
    T = np.eye(4)
    for spec in link_specs:
        T_step = _dh(**spec)
        T = T @ T_step
        transforms.append(T.copy())
    return transforms


# ── Per-robot FK ──────────────────────────────────────────────────────────────

def psm_fk(joints):
    """PSM (Large Needle Driver) — 7 joints, Modified DH."""
    j = list(joints) + [0] * (7 - len(joints))
    L_rcc          = 0.4389
    L_tool         = 0.416
    L_pitch2yaw    = 0.009
    L_yaw2ctrlpnt  = 0.0106
    specs = [
        dict(alpha= PI_2, a=0,           theta=j[0], d=0,             offset= PI_2,  joint_type='R', convention='MODIFIED'),
        dict(alpha=-PI_2, a=0,           theta=j[1], d=0,             offset=-PI_2,  joint_type='R', convention='MODIFIED'),
        dict(alpha= PI_2, a=0,           theta=0,    d=j[2],          offset=-L_rcc, joint_type='P', convention='MODIFIED'),
        dict(alpha=0,     a=0,           theta=j[3], d=L_tool,        offset=0,      joint_type='R', convention='MODIFIED'),
        dict(alpha=-PI_2, a=0,           theta=j[4], d=0,             offset=-PI_2,  joint_type='R', convention='MODIFIED'),
        dict(alpha=-PI_2, a=L_pitch2yaw, theta=j[5], d=0,             offset=-PI_2,  joint_type='R', convention='MODIFIED'),
        dict(alpha=-PI_2, a=0,           theta=0,    d=L_yaw2ctrlpnt, offset= PI_2,  joint_type='R', convention='MODIFIED'),
    ]
    return _chain(specs)


def ecm_fk(joints):
    """ECM (Endoscope Camera Manipulator) — 4 joints, Modified DH."""
    j = list(joints) + [0] * (4 - len(joints))
    L_rcc      = 0.3822
    L_scopelen = 0.385495
    specs = [
        dict(alpha= PI_2, a=0, theta=j[0], d=0,          offset= PI_2,  joint_type='R', convention='MODIFIED'),
        dict(alpha=-PI_2, a=0, theta=j[1], d=0,          offset=-PI_2,  joint_type='R', convention='MODIFIED'),
        dict(alpha= PI_2, a=0, theta=0,    d=j[2],       offset=-L_rcc, joint_type='P', convention='MODIFIED'),
        dict(alpha=0,     a=0, theta=j[3], d=L_scopelen, offset=0,      joint_type='R', convention='MODIFIED'),
    ]
    return _chain(specs)


def mtm_fk(joints):
    """MTM (Master Tool Manipulator) — 7 joints, Standard DH."""
    j = list(joints) + [0] * (7 - len(joints))
    L_arm     = 0.278828
    L_forearm = 0.363867
    L_h       = 0.147733
    specs = [
        dict(alpha= PI_2, a=0,         theta=j[0], d=0,   offset=-PI_2, joint_type='R', convention='STANDARD'),
        dict(alpha=0,     a=L_arm,     theta=j[1], d=0,   offset=-PI_2, joint_type='R', convention='STANDARD'),
        dict(alpha=-PI_2, a=L_forearm, theta=j[2], d=0,   offset= PI_2, joint_type='R', convention='STANDARD'),
        dict(alpha= PI_2, a=0,         theta=j[3], d=L_h, offset=0,     joint_type='R', convention='STANDARD'),
        dict(alpha=-PI_2, a=0,         theta=j[4], d=0,   offset=0,     joint_type='R', convention='STANDARD'),
        dict(alpha= PI_2, a=0,         theta=j[5], d=0,   offset=-PI_2, joint_type='R', convention='STANDARD'),
        dict(alpha=0,     a=0,         theta=j[6], d=0,   offset= PI_2, joint_type='R', convention='STANDARD'),
    ]
    return _chain(specs)


def raven2_fk(joints):
    """Raven2 — placeholder. Accurate DH params needed from mentor."""
    raise NotImplementedError(
        "Raven2 DH parameters not yet available. "
        "Please provide them from the Raven2 documentation."
    )


def galen_fk(joints):
    """
    Galen EE FK — 5 DOF: c1,c2,c3 (m), roll (rad), tilt (rad).
    Nominal kinematics from 20260414_galenEE.yaml (parallel stage linearised).
    Bernstein-polynomial T2R3 calibration applied at the tip when
    experiment/calibration/coef/bpoly_t.csv and bpoly_r.csv are present.
    """
    j = list(joints) + [0.0] * (5 - len(joints))
    c1, c2, c3, roll, tilt = j[0], j[1], j[2], j[3], j[4]

    R_c = 0.255   # carriage radius (m) — from YAML geometry
    h0  = 0.837   # nominal platform height from base at zero joints (m)

    # --- Mobile Platform (parallel stage) ---
    c_avg = (c1 + c2 + c3) / 3.0
    z_mp  = h0 + c_avg
    # Carriage unit vectors in Base XY: n1=(-1,0), n2=(0.5,-√3/2), n3=(0.5,+√3/2)
    plt_pitch = (c1 - 0.5*c2 - 0.5*c3) / R_c    # small rotation about base y
    plt_roll  = (c3 - c2) * (3**0.5 / 2) / R_c  # small rotation about base x
    cp, sp = np.cos(plt_pitch), np.sin(plt_pitch)
    crp, srp = np.cos(plt_roll), np.sin(plt_roll)
    R_mp = np.array([[cp, sp*srp, sp*crp],
                     [0,     crp,    -srp],
                     [-sp, cp*srp, cp*crp]])
    T_mp = np.eye(4)
    T_mp[:3, :3] = R_mp
    T_mp[:3,  3] = [0.0, 0.0, z_mp]

    # --- Roll Arm Base: pivot offset (0.031, 0, 0.058) in MP frame, then roll about z ---
    cr, sr = np.cos(roll), np.sin(roll)
    R_roll = np.array([[cr, -sr, 0],
                       [sr,  cr, 0],
                       [0,   0,  1]])
    T_rab = np.eye(4)
    T_rab[:3, :3] = R_mp @ R_roll
    T_rab[:3,  3] = T_mp[:3, :3] @ np.array([0.031, 0.0, 0.058]) + T_mp[:3, 3]

    # --- Tilt Distal: 0.588 m along Roll Arm z, then tilt about Roll Arm y ---
    tilt_eff = tilt - 0.126   # kinematic offset from YAML (offset: -0.12595)
    ct, st = np.cos(tilt_eff), np.sin(tilt_eff)
    R_tilt_j = np.array([[ct, 0,  st],
                          [0,  1,   0],
                          [-st, 0, ct]])
    T_tilt = np.eye(4)
    T_tilt[:3, :3] = T_rab[:3, :3] @ R_tilt_j
    T_tilt[:3,  3] = T_rab[:3, :3] @ np.array([0.0, 0.0, 0.588]) + T_rab[:3, 3]

    # --- Tip (dovetail): 0.032 m fixed along Tilt Distal z ---
    T_tip = np.eye(4)
    T_tip[:3, :3] = T_tilt[:3, :3]
    T_tip[:3,  3] = T_tilt[:3, :3] @ np.array([0.0, 0.0, 0.032]) + T_tilt[:3, 3]

    # --- Bernstein calibration corrections (T2R3 model, optical tracker data) ---
    bp_t, bp_r = _get_bpoly()
    if bp_t is not None and bp_r is not None:
        from scipy.spatial.transform import Rotation as _Rot
        _rX = _Rot.from_euler('XYZ', [roll, 0.0, 0.0])
        _rY = _Rot.from_euler('XYZ', [0.0, tilt, 0.0])
        delta_pos = T_tip[:3, 3] - (_D2R + _rX.apply(_R2T + _rY.apply(_T2F)))
        T_tip = T_tip @ _v6_htm(_bpoly_eval(delta_pos,           bp_t))
        T_tip = T_tip @ _v6_htm(_bpoly_eval(np.array([roll, tilt]), bp_r))

    return [T_mp, T_rab, T_tilt, T_tip]


# ── Node registry ─────────────────────────────────────────────────────────────

NODES = {
    "PSM": {
        "label":           "PSM  (Patient-Side Manipulator)",
        "n_joints":        7,
        "joint_names":     ["yaw", "pitch", "insertion", "tool_roll",
                            "tool_pitch", "tool_yaw", "gripper"],
        "joint_ranges":    [(-1.5, 1.5), (-1.5, 1.5), (0.0, 0.24),
                            (-3.0, 3.0), (-1.5, 1.5), (-1.5, 1.5), (-0.8, 0.8)],
        "fk":              psm_fk,
        "link_labels":     ["Link1", "Link2", "Insertion", "ToolRoll",
                            "ToolPitch", "ToolYaw", "Tip"],
        "ambf_namespace":  "/ambf/env/psm/",
        "ambf_base_link":  "baselink",
        "default_base_pos": [0.5,  0.5, -0.7],
        "default_sigma_joint":  0.001,    # rad — dynamic motion error
        "default_sigma_static": 0.001,    # rad — backlash / compliance floor
        "encoder_resolution":   0.00018,  # rad/count — 14-bit encoder over ±1.5 rad range
        "default_sigma_base":   0.002,    # m
        "color":           "#FF6B6B",
    },
    "ECM": {
        "label":           "ECM  (Endoscope Camera Manipulator)",
        "n_joints":        4,
        "joint_names":     ["yaw", "pitch", "insertion", "roll"],
        "joint_ranges":    [(-1.5, 1.5), (-1.5, 1.5), (0.0, 0.22), (-3.0, 3.0)],
        "fk":              ecm_fk,
        "link_labels":     ["Link1", "Link2", "Insertion", "Tip"],
        "ambf_namespace":  "/ambf/env/ecm/",
        "ambf_base_link":  "baselink",
        "default_base_pos": [0.5, -0.4, -0.6],
        "default_sigma_joint":  0.001,
        "default_sigma_static": 0.001,
        "encoder_resolution":   0.00018,  # rad/count — same 14-bit encoder as PSM
        "default_sigma_base":   0.002,
        "color":           "#4ECDC4",
    },
    "MTM": {
        "label":           "MTM  (Master Tool Manipulator)",
        "n_joints":        7,
        "joint_names":     ["shoulder_yaw", "shoulder_pitch", "elbow",
                            "wrist_platform", "wrist_pitch", "wrist_yaw", "jaw"],
        "joint_ranges":    [(-1.5, 1.5)] * 7,
        "fk":              mtm_fk,
        "link_labels":     ["Shoulder", "UpperArm", "Elbow",
                            "WristPlat", "WristPitch", "WristYaw", "Tip"],
        "ambf_namespace":  "/ambf/env/mtm/",
        "ambf_base_link":  "baselink",
        "default_base_pos": [-0.5, 0.0,  0.0],
        "default_sigma_joint":  0.0005,
        "default_sigma_static": 0.0005,
        "encoder_resolution":   0.00018,  # rad/count — same dVRK encoder family
        "default_sigma_base":   0.001,
        "color":           "#FFB347",
    },
    "Galen": {
        "label":           "Galen EE  (Surgical Endoscope Manipulator)",
        "n_joints":        5,
        "joint_names":     ["carriage1", "carriage2", "carriage3", "roll", "tilt"],
        "joint_ranges":    [(-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3),
                            (-1.5708, 1.5708), (-1.0472, 1.0472)],
        "fk":              galen_fk,
        "link_labels":     ["MobilePlatform", "RollArmBase", "TiltDistal", "Tip"],
        "ambf_namespace":  "/ambf/env/",
        "ambf_base_link":  "Base",
        "default_base_pos": [0.0, 0.0, -0.84],
        "default_sigma_joint":  0.001,
        "default_sigma_static": 0.001,
        "encoder_resolution":   0.00018,
        "default_sigma_base":   0.005,
        "color":           "#7EC8E3",
    },
    "Raven2": {
        "label":           "Raven2  (⚠ DH params pending)",
        "n_joints":        7,
        "joint_names":     ["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
        "joint_ranges":    [(-1.5, 1.5)] * 7,
        "fk":              raven2_fk,
        "link_labels":     ["L1", "L2", "L3", "L4", "L5", "L6", "Tip"],
        "ambf_namespace":  "/ambf/env/raven2/",
        "ambf_base_link":  "baselink",
        "default_base_pos": [-0.5, -0.5,  0.0],
        "default_sigma_joint":  0.001,
        "default_sigma_static": 0.002,
        "encoder_resolution":   0.00020,  # rad/count — placeholder, needs Raven2 datasheet
        "default_sigma_base":   0.002,
        "color":           "#DDA0DD",
    },
}
