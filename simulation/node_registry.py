"""
Node registry for surgical robotics uncertainty GUI.

Each entry defines a robot's kinematic chain (DH parameters),
tip link, default sigma values, and AMBF namespace for future
live connection.

DH sources:
  PSM / ECM / MTM  — ambf/core/ambf_controller/dvrk/scripts/
  Raven2           — placeholder, needs accurate DH from mentor
"""

import numpy as np

PI   = np.pi
PI_2 = np.pi / 2


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
        "default_sigma_joint": 0.001,   # rad (or m for prismatic)
        "default_sigma_base":  0.002,   # m
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
        "default_sigma_joint": 0.001,
        "default_sigma_base":  0.002,
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
        "default_sigma_joint": 0.0005,
        "default_sigma_base":  0.001,
        "color":           "#FFB347",
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
        "default_sigma_joint": 0.001,
        "default_sigma_base":  0.002,
        "color":           "#DDA0DD",
    },
}
