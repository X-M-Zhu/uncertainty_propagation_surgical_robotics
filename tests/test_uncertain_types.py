"""
Tests for uncertain_types.py — Dr. Taylor's CIS I uncertain Cartesian API.

Covers every operator overload listed in the spec:
  uPoint  : construction, addition
  uRot    : construction, applied to point, composition
  uFrame  : construction variants, composition, point transform
"""

import math
import numpy as np
import pytest
from uncertainty_networks import Point, Rot, Frame, uVector, uPoint, uRot, uFrame


# ── helpers ───────────────────────────────────────────────────────────────────

def _cov3(scale=1e-4):
    return scale * np.eye(3)

def _cov6(scale=1e-4):
    return scale * np.eye(6)


# ── uVector ───────────────────────────────────────────────────────────────────

def test_uvector_default_zero_covariance():
    uv = uVector([1.0, 2.0, 3.0])
    assert np.allclose(uv.C, np.zeros((3, 3)))
    assert np.allclose(uv.v, [1, 2, 3])


def test_uvector_explicit_covariance():
    uv = uVector([0.5], [[0.01]])
    assert uv.C.shape == (1, 1)
    assert math.isclose(uv.C[0, 0], 0.01)


# ── uPoint ────────────────────────────────────────────────────────────────────

def test_upoint_from_point():
    up = uPoint(Point(1.0, 2.0, 3.0), _cov3())
    assert math.isclose(up.x, 1.0)
    assert math.isclose(up.y, 2.0)
    assert math.isclose(up.z, 3.0)
    assert up.C.shape == (3, 3)


def test_upoint_from_array():
    up = uPoint([4.0, 5.0, 6.0], _cov3())
    assert math.isclose(up.x, 4.0)


def test_upoint_default_zero_covariance():
    up = uPoint(Point(0.0, 0.0, 0.0))
    assert np.allclose(up.C, np.zeros((3, 3)))


def test_upoint_add_two_upoints():
    up1 = uPoint(Point(1.0, 0.0, 0.0), _cov3(1e-4))
    up2 = uPoint(Point(0.0, 2.0, 0.0), _cov3(2e-4))
    up3 = up1 + up2
    assert math.isclose(up3.x, 1.0)
    assert math.isclose(up3.y, 2.0)
    assert np.allclose(up3.C, _cov3(1e-4) + _cov3(2e-4))


def test_upoint_add_nominal_plus_uncertain():
    p1   = Point(1.0, 0.0, 0.0)
    up2  = uPoint(Point(0.0, 1.0, 0.0), _cov3(5e-4))
    up3a = p1 + up2      # Point + uPoint  (__radd__)
    up3b = up2 + p1      # uPoint + Point  (__add__)
    for up3 in (up3a, up3b):
        assert math.isclose(up3.x, 1.0)
        assert math.isclose(up3.y, 1.0)
        assert np.allclose(up3.C, _cov3(5e-4))


def test_upoint_add_covariance_grows():
    up1 = uPoint(Point(0, 0, 0), _cov3(1e-4))
    up2 = uPoint(Point(0, 0, 0), _cov3(1e-4))
    up3 = up1 + up2
    assert np.all(np.diag(up3.C) >= np.diag(up1.C))


# ── uRot ─────────────────────────────────────────────────────────────────────

def test_urot_from_rot():
    R  = Rot(axis='z', angle=math.pi / 4)
    uR = uRot(R, _cov3())
    assert np.allclose(uR.matrix, R.matrix)
    assert np.allclose(uR.C, _cov3())


def test_urot_default_zero_covariance():
    uR = uRot(Rot(axis='x', angle=0.1))
    assert np.allclose(uR.C, np.zeros((3, 3)))


def test_urot_from_axis_uangle():
    uangle = uVector([math.pi / 2], [[1e-4]])
    uR = uRot('z', uangle)
    expected_R = Rot(axis='z', angle=math.pi / 2)
    np.testing.assert_allclose(uR.matrix, expected_R.matrix, atol=1e-7)
    # Covariance must be non-zero (propagated from angle uncertainty)
    assert np.any(uR.C > 0)


def test_urot_from_uaxis_angle():
    uaxis  = uVector([0.0, 0.0, 1.0], _cov3(1e-3))
    uR = uRot(uaxis, math.pi / 4)
    assert np.any(uR.C > 0)


def test_urot_from_upoint_axis_angle():
    uaxis = uPoint(Point(0.0, 0.0, 1.0), _cov3(1e-3))
    uR = uRot(uaxis, math.pi / 4)
    assert np.any(uR.C > 0)
    expected = Rot(axis='z', angle=math.pi / 4)
    np.testing.assert_allclose(uR.matrix, expected.matrix, atol=1e-7)


def test_urot_from_uaxis_uangle():
    uaxis  = uVector([0.0, 0.0, 1.0], _cov3(1e-3))
    uangle = uVector([math.pi / 4], [[1e-4]])
    uR = uRot(uaxis, uangle)
    # Both sources contribute — covariance must be larger than from axis alone
    uR_axis_only = uRot(uaxis, math.pi / 4)
    assert np.trace(uR.C) > np.trace(uR_axis_only.C)
    # Nominal rotation should match the nominal angle
    expected = Rot(axis='z', angle=math.pi / 4)
    np.testing.assert_allclose(uR.matrix, expected.matrix, atol=1e-7)


def test_urot_times_point():
    uR = uRot(Rot(axis='z', angle=math.pi / 2), _cov3(1e-4))
    p  = Point(1.0, 0.0, 0.0)
    up = uR * p
    assert isinstance(up, uPoint)
    assert math.isclose(up.x,  0.0, abs_tol=1e-7)
    assert math.isclose(up.y,  1.0, abs_tol=1e-7)
    assert np.any(up.C > 0)


def test_urot_times_upoint():
    uR  = uRot(Rot(axis='z', angle=0.0), _cov3(1e-4))
    up1 = uPoint(Point(1.0, 0.0, 0.0), _cov3(1e-4))
    up2 = uR * up1
    assert isinstance(up2, uPoint)
    # Covariance should be larger than either input alone
    assert np.trace(up2.C) > np.trace(up1.C)


def test_urot_compose_two_urot():
    uR1 = uRot(Rot(axis='z', angle=math.pi / 4), _cov3(1e-4))
    uR2 = uRot(Rot(axis='z', angle=math.pi / 4), _cov3(1e-4))
    uR3 = uR1 * uR2
    expected = Rot(axis='z', angle=math.pi / 2)
    np.testing.assert_allclose(uR3.matrix, expected.matrix, atol=1e-7)
    assert np.all(np.diag(uR3.C) >= np.diag(uR1.C))


def test_urot_compose_rot_times_urot():
    R1  = Rot(axis='z', angle=math.pi / 4)
    uR2 = uRot(Rot(axis='z', angle=math.pi / 4), _cov3(1e-4))
    uR3 = R1 * uR2
    assert isinstance(uR3, uRot)
    # Nominal should match composed rotation
    expected = Rot(axis='z', angle=math.pi / 2)
    np.testing.assert_allclose(uR3.matrix, expected.matrix, atol=1e-7)
    # Covariance stays the same (R1 adds no uncertainty)
    np.testing.assert_allclose(uR3.C, uR2.C)


def test_urot_compose_urot_times_rot():
    uR1 = uRot(Rot(axis='z', angle=math.pi / 4), _cov3(1e-4))
    R2  = Rot(axis='z', angle=math.pi / 4)
    uR3 = uR1 * R2
    assert isinstance(uR3, uRot)
    expected = Rot(axis='z', angle=math.pi / 2)
    np.testing.assert_allclose(uR3.matrix, expected.matrix, atol=1e-7)


# ── uFrame ────────────────────────────────────────────────────────────────────

def _make_frame(dx=0.1, dy=0.0, dz=0.0, axis='z', angle=0.0):
    return Frame(Rot(axis=axis, angle=angle), Point(dx, dy, dz))


def test_uframe_from_frame():
    F  = _make_frame(dx=0.5)
    uF = uFrame(F, _cov6())
    assert np.allclose(uF.F_nom[:3, 3], [0.5, 0.0, 0.0])
    assert np.allclose(uF.C, _cov6())


def test_uframe_default_zero_covariance():
    uF = uFrame(_make_frame())
    assert np.allclose(uF.C, np.zeros((6, 6)))


def test_uframe_from_urot_upoint():
    uR = uRot(Rot(axis='z', angle=0.0), _cov3(1e-4))
    up = uPoint(Point(0.1, 0.0, 0.0), _cov3(2e-4))
    uF = uFrame(uR, up)
    np.testing.assert_allclose(uF.C[:3, :3], _cov3(1e-4))
    np.testing.assert_allclose(uF.C[3:, 3:], _cov3(2e-4))


def test_uframe_from_rot_upoint():
    R  = Rot(axis='z', angle=0.0)
    up = uPoint(Point(0.2, 0.0, 0.0), _cov3(3e-4))
    uF = uFrame(R, up)
    np.testing.assert_allclose(uF.C[:3, :3], np.zeros((3, 3)))
    np.testing.assert_allclose(uF.C[3:, 3:], _cov3(3e-4))


def test_uframe_from_urot_point():
    uR = uRot(Rot(axis='z', angle=0.0), _cov3(5e-4))
    p  = Point(0.3, 0.0, 0.0)
    uF = uFrame(uR, p)
    np.testing.assert_allclose(uF.C[:3, :3], _cov3(5e-4))
    np.testing.assert_allclose(uF.C[3:, 3:], np.zeros((3, 3)))


def test_uframe_times_uframe_covariance_grows():
    uF1 = uFrame(_make_frame(dx=0.1), _cov6(1e-6))
    uF2 = uFrame(_make_frame(dx=0.1), _cov6(1e-6))
    uF3 = uF1 * uF2
    assert np.all(np.diag(uF3.C) >= np.diag(uF1.C))


def test_uframe_times_uframe_nominal():
    uF1 = uFrame(_make_frame(dx=0.3))
    uF2 = uFrame(_make_frame(dx=0.2))
    uF3 = uF1 * uF2
    np.testing.assert_allclose(uF3.F_nom[:3, 3], [0.5, 0.0, 0.0], atol=1e-9)


def test_uframe_times_frame():
    uF1 = uFrame(_make_frame(dx=0.1), _cov6(1e-6))
    F2  = _make_frame(dx=0.2)
    uF3 = uF1 * F2
    assert isinstance(uF3, uFrame)
    np.testing.assert_allclose(uF3.F_nom[:3, 3], [0.3, 0.0, 0.0], atol=1e-9)


def test_frame_times_uframe():
    F1  = _make_frame(dx=0.1)
    uF2 = uFrame(_make_frame(dx=0.2), _cov6(1e-6))
    uF3 = F1 * uF2
    assert isinstance(uF3, uFrame)
    np.testing.assert_allclose(uF3.F_nom[:3, 3], [0.3, 0.0, 0.0], atol=1e-9)


def test_uframe_times_upoint():
    uF  = uFrame(_make_frame(dx=0.5), _cov6(1e-6))
    up1 = uPoint(Point(0.0, 0.0, 0.0), _cov3(1e-6))
    up2 = uF * up1
    assert isinstance(up2, uPoint)
    assert math.isclose(up2.x, 0.5, abs_tol=1e-9)
    assert np.any(up2.C > 0)


def test_uframe_times_point():
    uF = uFrame(_make_frame(dx=0.5), _cov6(1e-6))
    p  = Point(0.1, 0.0, 0.0)
    up = uF * p
    assert isinstance(up, uPoint)
    assert math.isclose(up.x, 0.6, abs_tol=1e-9)
    assert np.any(up.C > 0)


def test_uframe_matmul_alias():
    uF1 = uFrame(_make_frame(dx=0.1), _cov6(1e-6))
    uF2 = uFrame(_make_frame(dx=0.2), _cov6(1e-6))
    uF3a = uF1 * uF2
    uF3b = uF1 @ uF2
    np.testing.assert_allclose(uF3a.F_nom, uF3b.F_nom)


def test_rot_times_uframe():
    R1  = Rot(axis='z', angle=math.pi / 2)
    uF2 = uFrame(_make_frame(dx=0.1), _cov6(1e-6))
    uF3 = R1 * uF2
    assert isinstance(uF3, uFrame)
    # R1 rotates the translation of uF2: (0.1,0,0) -> (0, 0.1, 0)
    np.testing.assert_allclose(uF3.F_nom[:3, 3], [0.0, 0.1, 0.0], atol=1e-7)
    # Covariance unchanged (R1 is certain)
    np.testing.assert_allclose(uF3.C, uF2.C)


def test_uframe_times_rot():
    uF1 = uFrame(_make_frame(dx=0.1), _cov6(1e-6))
    R2  = Rot(axis='z', angle=math.pi / 2)
    uF3 = uF1 * R2
    assert isinstance(uF3, uFrame)
    # Translation stays the same; rotation changes
    np.testing.assert_allclose(uF3.F_nom[:3, 3], [0.1, 0.0, 0.0], atol=1e-7)


def test_uframe_inv():
    uF  = uFrame(_make_frame(dx=0.5), _cov6(1e-6))
    uFi = uF.inv()
    uI  = uF * uFi
    np.testing.assert_allclose(uI.F_nom, np.eye(4), atol=1e-9)
