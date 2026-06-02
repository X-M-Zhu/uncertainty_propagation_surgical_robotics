"""
Tests for Galen EE forward kinematics (galen_fk in node_registry).

Galen uses a parallel carriage mechanism (not DH parameters), so tests
focus on geometric invariants rather than DH identities.

Geometry reference
------------------
- h0 = 0.837 m  : nominal platform height at zero joints
- R_c = 0.255 m : carriage radius
- Roll pivot offset from MP origin: (0.031, 0, 0.058) m
- Tilt arm length: 0.588 m along roll-arm z
- Tip offset: 0.032 m along tilt-distal z
- Tilt kinematic offset: -0.12595 rad (from YAML)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulation'))

import numpy as np
import pytest
from node_registry import galen_fk
from uncertainty_networks.se3 import is_se3


# -- helpers ------------------------------------------------------------------

LABELS = ["MobilePlatform", "RollArmBase", "TiltDistal", "Tip"]
ZERO   = [0.0, 0.0, 0.0, 0.0, 0.0]
H0     = 0.837   # nominal platform height (m)
R_C    = 0.255   # carriage radius (m)


def _tip(joints):
    return galen_fk(joints)[-1][:3, 3]


# -- basic structure ----------------------------------------------------------

def test_returns_four_transforms():
    transforms = galen_fk(ZERO)
    assert len(transforms) == 4


def test_all_transforms_valid_se3():
    for joints in [ZERO, [0.05, 0.05, 0.05, 0, 0], [0, 0, 0, 0.3, 0.2]]:
        for T in galen_fk(joints):
            assert is_se3(T), f"Not a valid SE(3) transform for joints={joints}"


def test_output_shapes():
    for T in galen_fk(ZERO):
        assert T.shape == (4, 4)


# -- zero-joint geometry ------------------------------------------------------

def test_platform_height_at_zero_joints():
    T_mp = galen_fk(ZERO)[0]
    assert abs(T_mp[2, 3] - H0) < 1e-9, \
        f"Platform z at zero joints should be h0={H0}, got {T_mp[2,3]}"


def test_platform_identity_rotation_at_zero_joints():
    """Equal carriages -> no platform tilt."""
    T_mp = galen_fk(ZERO)[0]
    assert np.allclose(T_mp[:3, :3], np.eye(3), atol=1e-9)


def test_roll_arm_pivot_offset_at_zero():
    """Roll arm base should be offset (0.031, 0, 0.058) from platform at zero."""
    T_mp  = galen_fk(ZERO)[0]
    T_rab = galen_fk(ZERO)[1]
    delta = T_rab[:3, 3] - T_mp[:3, 3]
    expected = np.array([0.031, 0.0, 0.058])
    assert np.allclose(delta, expected, atol=1e-6), \
        f"Roll arm pivot offset wrong: got {delta}, expected {expected}"


# -- carriage (parallel stage) behaviour --------------------------------------

def test_equal_carriage_raise_lifts_all_frames():
    """All three carriages +delta -> all frame z positions rise by delta."""
    delta = 0.05
    T0 = galen_fk(ZERO)
    T1 = galen_fk([delta, delta, delta, 0, 0])
    for i in range(4):
        dz = T1[i][2, 3] - T0[i][2, 3]
        assert abs(dz - delta) < 1e-6, \
            f"{LABELS[i]} z should rise by {delta}, got {dz:.6f}"


def test_equal_carriage_raise_no_rotation():
    """Equal carriage movement should not tilt the platform."""
    T_mp = galen_fk([0.05, 0.05, 0.05, 0, 0])[0]
    assert np.allclose(T_mp[:3, :3], np.eye(3), atol=1e-6)


def test_c1_only_tilts_platform_about_y():
    """c1 > 0 with c2=c3=0 should tilt the platform (non-identity rotation)."""
    T_mp = galen_fk([0.05, 0, 0, 0, 0])[0]
    assert not np.allclose(T_mp[:3, :3], np.eye(3), atol=1e-3), \
        "c1-only carriage should produce a platform tilt"


def test_c2_c3_differential_tilts_platform():
    """c2 != c3 should produce a roll tilt of the platform."""
    T_mp = galen_fk([0, 0.05, -0.05, 0, 0])[0]
    assert not np.allclose(T_mp[:3, :3], np.eye(3), atol=1e-3)


# -- roll joint ---------------------------------------------------------------

def test_roll_zero_gives_identity_like_rotation():
    """At zero roll the roll-arm rotation should equal the platform rotation."""
    T_mp  = galen_fk(ZERO)[0]
    T_rab = galen_fk(ZERO)[1]
    assert np.allclose(T_rab[:3, :3], T_mp[:3, :3], atol=1e-9)


def test_roll_changes_orientation():
    T0 = galen_fk([0, 0, 0, 0,   0])[1]
    T1 = galen_fk([0, 0, 0, 0.5, 0])[1]
    assert not np.allclose(T0[:3, :3], T1[:3, :3], atol=1e-3)


def test_roll_does_not_change_z_height():
    """Roll is pure rotation around a fixed pivot -- z of roll arm base unchanged."""
    z0 = galen_fk([0, 0, 0, 0,   0])[1][2, 3]
    z1 = galen_fk([0, 0, 0, 0.5, 0])[1][2, 3]
    assert abs(z1 - z0) < 1e-6


# -- tilt joint ---------------------------------------------------------------

def test_tilt_changes_distal_rotation():
    """TiltDistal is the pivot -- position is fixed, only rotation changes."""
    T0 = galen_fk([0, 0, 0, 0, 0  ])[2]
    T1 = galen_fk([0, 0, 0, 0, 0.3])[2]
    # Position (pivot) must stay the same
    assert np.allclose(T0[:3, 3], T1[:3, 3], atol=1e-6), \
        "TiltDistal pivot position should not move with tilt joint"
    # Rotation must change
    assert not np.allclose(T0[:3, :3], T1[:3, :3], atol=1e-3), \
        "TiltDistal rotation should change with tilt joint"


def test_tilt_changes_tip_position():
    tip0 = _tip([0, 0, 0, 0, 0  ])
    tip1 = _tip([0, 0, 0, 0, 0.3])
    assert np.linalg.norm(tip1 - tip0) > 1e-3


# -- continuity ---------------------------------------------------------------

def test_tip_continuous_in_joints():
    """Small joint perturbation -> small tip displacement."""
    eps = 1e-4
    base_tip = _tip(ZERO)
    for i in range(5):
        j = ZERO.copy()
        j[i] += eps
        perturbed_tip = _tip(j)
        dist = np.linalg.norm(perturbed_tip - base_tip)
        assert dist < 0.01, \
            f"Joint {i} perturbation of {eps} moved tip by {dist:.4f} m -- too large"


# -- short-input padding ------------------------------------------------------

def test_accepts_fewer_than_5_joints():
    """galen_fk should pad missing joints with zero."""
    T_full  = galen_fk([0, 0, 0, 0, 0])
    T_short = galen_fk([0, 0, 0])
    assert np.allclose(T_full[-1], T_short[-1], atol=1e-12)
