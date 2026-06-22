"""
Tests for uncertain_types.py — Dr. Taylor's CIS I uncertain Cartesian API.

Covers every operator overload listed in the spec:
  uvct3  : construction, addition
  uRot    : construction, applied to point, composition
  uFrame  : construction variants, composition, point transform
"""

import math
import numpy as np
import pytest
from uncertainty_networks import vct3, Rot, Frame, uScalar, uVector, uvct3, uRot, uFrame


# ── helpers ───────────────────────────────────────────────────────────────────

def _cov3(scale=1e-4):
    return scale * np.eye(3)

def _cov6(scale=1e-4):
    return scale * np.eye(6)


# ── uScalar ───────────────────────────────────────────────────────────────────

def test_uscalar_construction():
    us = uScalar(3.0, 0.04)
    assert math.isclose(us.s,   3.0)
    assert math.isclose(us.var, 0.04)
    assert math.isclose(us.std, 0.2)


def test_uscalar_default_zero_variance():
    us = uScalar(5.0)
    assert math.isclose(us.var, 0.0)
    assert math.isclose(us.std, 0.0)


def test_uscalar_add_two():
    us3 = uScalar(1.0, 0.01) + uScalar(2.0, 0.04)
    assert math.isclose(us3.s,   3.0)
    assert math.isclose(us3.var, 0.05)


def test_uscalar_add_float():
    us2 = uScalar(1.0, 0.01) + 2.0
    assert math.isclose(us2.s,   3.0)
    assert math.isclose(us2.var, 0.01)   # constant shift adds no variance


def test_uscalar_radd_float():
    us2 = 2.0 + uScalar(1.0, 0.01)
    assert math.isclose(us2.s,   3.0)
    assert math.isclose(us2.var, 0.01)


def test_uscalar_sub():
    us3 = uScalar(5.0, 0.04) - uScalar(2.0, 0.01)
    assert math.isclose(us3.s,   3.0)
    assert math.isclose(us3.var, 0.05)   # variances add even for subtraction


def test_uscalar_sub_float():
    us2 = uScalar(5.0, 0.04) - 2.0
    assert math.isclose(us2.s,   3.0)
    assert math.isclose(us2.var, 0.04)


def test_uscalar_rsub_float():
    us2 = 5.0 - uScalar(2.0, 0.04)
    assert math.isclose(us2.s,   3.0)
    assert math.isclose(us2.var, 0.04)


def test_uscalar_neg():
    un = -uScalar(3.0, 0.09)
    assert math.isclose(un.s,   -3.0)
    assert math.isclose(un.var,  0.09)   # negation leaves variance unchanged


def test_uscalar_mul_float():
    us2 = 3.0 * uScalar(2.0, 0.04)
    assert math.isclose(us2.s,   6.0)
    assert math.isclose(us2.var, 9.0 * 0.04)


def test_uscalar_mul_two():
    # var(a*b) = b²*var_a + a²*var_b
    us3 = uScalar(2.0, 0.01) * uScalar(3.0, 0.04)
    assert math.isclose(us3.s,   6.0)
    assert math.isclose(us3.var, 3.0 ** 2 * 0.01 + 2.0 ** 2 * 0.04)


def test_uscalar_div_float():
    us2 = uScalar(6.0, 0.09) / 3.0
    assert math.isclose(us2.s,   2.0)
    assert math.isclose(us2.var, 0.09 / 9.0)


def test_uscalar_div_two():
    # d(a/b)/da = 1/b,  d(a/b)/db = -a/b²
    us1 = uScalar(6.0, 0.09)
    us2 = uScalar(3.0, 0.01)
    us3 = us1 / us2
    assert math.isclose(us3.s, 2.0)
    expected_var = 0.09 / 9.0 + (6.0 / 9.0) ** 2 * 0.01
    assert math.isclose(us3.var, expected_var)


def test_uscalar_rdiv_float():
    # f(s) = a/s  →  var_out = (a/s²)² * var
    us  = uScalar(2.0, 0.04)
    us2 = 6.0 / us
    assert math.isclose(us2.s, 3.0)
    assert math.isclose(us2.var, (6.0 / 4.0) ** 2 * 0.04)


def test_uscalar_pow():
    # f(s) = s²  →  df/ds = 2s  →  var_out = (2s)² * var
    us2 = uScalar(3.0, 0.01) ** 2
    assert math.isclose(us2.s,   9.0)
    assert math.isclose(us2.var, (2 * 3.0) ** 2 * 0.01)


# ── uVector ───────────────────────────────────────────────────────────────────

def test_uvector_default_zero_covariance():
    uv = uVector([1.0, 2.0, 3.0])
    assert np.allclose(uv.C, np.zeros((3, 3)))
    assert np.allclose(uv.v, [1, 2, 3])


def test_uvector_explicit_covariance():
    uv = uVector([0.5], [[0.01]])
    assert uv.C.shape == (1, 1)
    assert math.isclose(uv.C[0, 0], 0.01)


def test_uvector_add():
    uv3 = uVector([1.0, 0.0], np.diag([0.01, 0.02])) + uVector([0.0, 2.0], np.diag([0.03, 0.04]))
    assert np.allclose(uv3.v, [1.0, 2.0])
    assert np.allclose(uv3.C, np.diag([0.04, 0.06]))


def test_uvector_add_array():
    uv  = uVector([1.0, 2.0], np.eye(2) * 0.01)
    uv2 = uv + np.array([3.0, 4.0])
    assert np.allclose(uv2.v, [4.0, 6.0])
    assert np.allclose(uv2.C, np.eye(2) * 0.01)   # constant adds no variance


def test_uvector_radd_array():
    uv  = uVector([1.0, 2.0], np.eye(2) * 0.01)
    uv2 = np.array([3.0, 4.0]) + uv
    assert np.allclose(uv2.v, [4.0, 6.0])
    assert np.allclose(uv2.C, np.eye(2) * 0.01)


def test_uvector_sub():
    uv3 = uVector([3.0, 4.0], np.eye(2) * 0.02) - uVector([1.0, 1.0], np.eye(2) * 0.01)
    assert np.allclose(uv3.v, [2.0, 3.0])
    assert np.allclose(uv3.C, np.eye(2) * 0.03)   # covariances add


def test_uvector_sub_array():
    uv  = uVector([3.0, 4.0], np.eye(2) * 0.02)
    uv2 = uv - np.array([1.0, 1.0])
    assert np.allclose(uv2.v, [2.0, 3.0])
    assert np.allclose(uv2.C, np.eye(2) * 0.02)


def test_uvector_neg():
    uvn = -uVector([1.0, -2.0], np.eye(2) * 0.01)
    assert np.allclose(uvn.v, [-1.0, 2.0])
    assert np.allclose(uvn.C, np.eye(2) * 0.01)   # variance unchanged by negation


def test_uvector_scalar_mul():
    uv2 = 3.0 * uVector([1.0, 2.0], np.eye(2) * 0.01)
    assert np.allclose(uv2.v, [3.0, 6.0])
    assert np.allclose(uv2.C, np.eye(2) * 0.09)   # 3² * 0.01


def test_uvector_rmatmul_matrix_gives_uvector():
    # A @ uv: (m,n) @ (n,) -> uVector of dim m
    uv = uVector([1.0, 0.0], np.diag([0.04, 0.01]))
    A  = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])  # (3,2)
    result = A @ uv
    assert isinstance(result, uVector)
    assert np.allclose(result.v, A @ uv.v)
    assert np.allclose(result.C, A @ uv.C @ A.T)


def test_uvector_rmatmul_vector_gives_uscalar():
    # w @ uv: (n,) @ (n,) -> uScalar
    uv = uVector([1.0, 2.0], np.diag([0.01, 0.04]))
    w  = np.array([3.0, 4.0])
    result = w @ uv
    assert isinstance(result, uScalar)
    assert math.isclose(result.s,   float(w @ uv.v))
    assert math.isclose(result.var, float(w @ uv.C @ w))


def test_uvector_matmul_matrix_gives_uvector():
    # uv @ A: (n,) @ (n,m) -> uVector of dim m
    uv = uVector([1.0, 2.0], np.diag([0.01, 0.04]))
    A  = np.array([[1.0, 0.0], [0.0, 2.0]])  # (2,2)
    result = uv @ A
    assert isinstance(result, uVector)
    assert np.allclose(result.v, uv.v @ A)
    assert np.allclose(result.C, A.T @ uv.C @ A)


def test_uvector_dot():
    uv = uVector([1.0, 2.0], np.diag([0.01, 0.04]))
    w  = np.array([3.0, 4.0])
    us = uv.dot(w)
    assert isinstance(us, uScalar)
    assert math.isclose(us.s,   11.0)              # 3*1 + 4*2
    assert math.isclose(us.var, 9.0 * 0.01 + 16.0 * 0.04)


def test_uvector_norm():
    uv = uVector([3.0, 4.0], np.diag([0.01, 0.04]))
    us = uv.norm()
    assert isinstance(us, uScalar)
    assert math.isclose(us.s, 5.0)
    u = np.array([0.6, 0.8])                       # unit vector [3,4]/5
    assert math.isclose(us.var, float(u @ uv.C @ u))


# ── uvct3 ────────────────────────────────────────────────────────────────────

def test_upoint_from_point():
    up = uvct3(vct3(1.0, 2.0, 3.0), _cov3())
    assert math.isclose(up.x, 1.0)
    assert math.isclose(up.y, 2.0)
    assert math.isclose(up.z, 3.0)
    assert up.C.shape == (3, 3)


def test_upoint_from_array():
    up = uvct3([4.0, 5.0, 6.0], _cov3())
    assert math.isclose(up.x, 4.0)


def test_upoint_default_zero_covariance():
    up = uvct3(vct3(0.0, 0.0, 0.0))
    assert np.allclose(up.C, np.zeros((3, 3)))


def test_upoint_add_two_upoints():
    up1 = uvct3(vct3(1.0, 0.0, 0.0), _cov3(1e-4))
    up2 = uvct3(vct3(0.0, 2.0, 0.0), _cov3(2e-4))
    up3 = up1 + up2
    assert math.isclose(up3.x, 1.0)
    assert math.isclose(up3.y, 2.0)
    assert np.allclose(up3.C, _cov3(1e-4) + _cov3(2e-4))


def test_upoint_add_nominal_plus_uncertain():
    p1   = vct3(1.0, 0.0, 0.0)
    up2  = uvct3(vct3(0.0, 1.0, 0.0), _cov3(5e-4))
    up3a = p1 + up2      # Point + uvct3  (__radd__)
    up3b = up2 + p1      # uvct3 + Point  (__add__)
    for up3 in (up3a, up3b):
        assert math.isclose(up3.x, 1.0)
        assert math.isclose(up3.y, 1.0)
        assert np.allclose(up3.C, _cov3(5e-4))


def test_upoint_add_covariance_grows():
    up1 = uvct3(vct3(0, 0, 0), _cov3(1e-4))
    up2 = uvct3(vct3(0, 0, 0), _cov3(1e-4))
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


def test_urot_from_axis_uscalar():
    """uRot('z', uScalar) should give same result as uRot('z', uVector([θ],[[var]]))."""
    angle = math.pi / 2
    var   = 1e-4
    uR_scalar = uRot('z', uScalar(angle, var))
    uR_vector = uRot('z', uVector([angle], [[var]]))
    np.testing.assert_allclose(uR_scalar.matrix, uR_vector.matrix, atol=1e-12)
    np.testing.assert_allclose(uR_scalar.C,      uR_vector.C,      atol=1e-14)


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
    uaxis = uvct3(vct3(0.0, 0.0, 1.0), _cov3(1e-3))
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


def test_urot_from_uaxis_uscalar():
    """uRot(uAxis, uScalar) should match uRot(uAxis, uVector([θ],[[var]]))."""
    angle  = math.pi / 4
    var    = 1e-4
    uaxis  = uVector([0.0, 0.0, 1.0], _cov3(1e-3))
    uR_sc  = uRot(uaxis, uScalar(angle, var))
    uR_vec = uRot(uaxis, uVector([angle], [[var]]))
    np.testing.assert_allclose(uR_sc.matrix, uR_vec.matrix, atol=1e-12)
    np.testing.assert_allclose(uR_sc.C,      uR_vec.C,      atol=1e-14)


def test_urot_times_point():
    uR = uRot(Rot(axis='z', angle=math.pi / 2), _cov3(1e-4))
    p  = vct3(1.0, 0.0, 0.0)
    up = uR * p
    assert isinstance(up, uvct3)
    assert math.isclose(up.x,  0.0, abs_tol=1e-7)
    assert math.isclose(up.y,  1.0, abs_tol=1e-7)
    assert np.any(up.C > 0)


def test_urot_times_upoint():
    uR  = uRot(Rot(axis='z', angle=0.0), _cov3(1e-4))
    up1 = uvct3(vct3(1.0, 0.0, 0.0), _cov3(1e-4))
    up2 = uR * up1
    assert isinstance(up2, uvct3)
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
    return Frame(Rot(axis=axis, angle=angle), vct3(dx, dy, dz))


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
    up = uvct3(vct3(0.1, 0.0, 0.0), _cov3(2e-4))
    uF = uFrame(uR, up)
    np.testing.assert_allclose(uF.C[:3, :3], _cov3(1e-4))
    np.testing.assert_allclose(uF.C[3:, 3:], _cov3(2e-4))


def test_uframe_from_rot_upoint():
    R  = Rot(axis='z', angle=0.0)
    up = uvct3(vct3(0.2, 0.0, 0.0), _cov3(3e-4))
    uF = uFrame(R, up)
    np.testing.assert_allclose(uF.C[:3, :3], np.zeros((3, 3)))
    np.testing.assert_allclose(uF.C[3:, 3:], _cov3(3e-4))


def test_uframe_from_urot_point():
    uR = uRot(Rot(axis='z', angle=0.0), _cov3(5e-4))
    p  = vct3(0.3, 0.0, 0.0)
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
    up1 = uvct3(vct3(0.0, 0.0, 0.0), _cov3(1e-6))
    up2 = uF * up1
    assert isinstance(up2, uvct3)
    assert math.isclose(up2.x, 0.5, abs_tol=1e-9)
    assert np.any(up2.C > 0)


def test_uframe_times_point():
    uF = uFrame(_make_frame(dx=0.5), _cov6(1e-6))
    p  = vct3(0.1, 0.0, 0.0)
    up = uF * p
    assert isinstance(up, uvct3)
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
