"""
Builds a GeometricNetwork from the user's selected nodes and their
configuration (base positions, sigma values, parent connections).

Works in two modes:
  - mock  : joint angles driven by sine waves (runs on macOS, no AMBF needed)
  - live  : joint angles read from AMBF Python client (requires ROS + AMBF)
"""

import sys
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from uncertainty_networks import GeometricNetwork, UncertainTransform

from node_registry import NODES


def _make_uncertain_transform(F_nom, sigma_rot, sigma_trans):
    """4x4 nominal transform + isotropic 6x6 covariance (rot, trans ordering)."""
    C = np.diag([sigma_rot**2] * 3 + [sigma_trans**2] * 3)
    return UncertainTransform(F_nom, C)


def _base_transform(pos):
    """Fixed base-to-World transform for a robot at position pos=[x,y,z]."""
    T = np.eye(4)
    T[:3, 3] = pos
    return T


def build_network(selections):
    """
    Build a GeometricNetwork from the GUI selections.

    Parameters
    ----------
    selections : list of dicts, each with keys:
        name          - robot name  (e.g. "PSM")
        joint_angles  - np.array of current joint positions
        sigma_joint   - float, noise std per joint (rad or m)
        sigma_base    - float, base registration noise (m)
        base_pos      - [x, y, z] of robot base in World frame

    Returns
    -------
    net       : GeometricNetwork
    tip_nodes : dict  name -> tip frame label  (e.g. "PSM_Tip")
    """
    net = GeometricNetwork()

    tip_nodes = {}
    for sel in selections:
        name    = sel["name"]
        node    = NODES[name]
        joints  = sel["joint_angles"]
        sj      = sel["sigma_joint"]
        sb      = sel["sigma_base"]
        base_pos = sel["base_pos"]

        # World → RobotBase (base registration uncertainty)
        T_base = _base_transform(base_pos)
        base_frame = f"{name}_Base"
        net.add_edge("World", base_frame,
                     _make_uncertain_transform(T_base, sb, sb))

        # RobotBase → Link_k  (kinematic chain)
        try:
            transforms = node["fk"](joints)
        except NotImplementedError:
            continue

        labels = node["link_labels"]
        prev_T = np.eye(4)
        prev_frame = base_frame

        for k, (T_k_0, label) in enumerate(zip(transforms, labels)):
            cur_frame = f"{name}_{label}"
            T_step = np.linalg.inv(prev_T) @ T_k_0
            net.add_edge(prev_frame, cur_frame,
                         _make_uncertain_transform(T_step, sj, sj))
            prev_T = T_k_0
            prev_frame = cur_frame

        tip_nodes[name] = f"{name}_{labels[-1]}"

    return net, tip_nodes


def mock_joint_trajectory(t, name):
    """
    Generate time-varying joint angles for a robot using sine waves.
    Each joint oscillates at a slightly different frequency so the
    motion looks natural and non-repetitive.
    """
    node = NODES[name]
    n    = node["n_joints"]
    lo   = np.array([r[0] for r in node["joint_ranges"]])
    hi   = np.array([r[1] for r in node["joint_ranges"]])
    mid  = (lo + hi) / 2
    amp  = (hi - lo) / 4          # use quarter-range so motion is moderate

    freqs = 0.3 + 0.1 * np.arange(n)   # 0.3, 0.4, 0.5, ... Hz
    angles = mid + amp * np.sin(2 * np.pi * freqs * t)
    return np.clip(angles, lo, hi)
