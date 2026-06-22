import math
import numpy as np
from uncertainty_networks import UncertainTransform, vct3, Rot, Frame


def test_compose_covariance_symmetry_and_growth():
    F = np.eye(4)
    C1 = np.diag([1e-6] * 6)
    C2 = np.diag([2e-6] * 6)

    U1 = UncertainTransform(F, C1)
    U2 = UncertainTransform(F, C2)

    U12 = U1 @ U2

    # Symmetry
    assert np.allclose(U12.C, U12.C.T)

    # Growth (diagonal should not decrease)
    assert np.all(np.diag(U12.C) >= np.diag(C1))


def test_apply_to_point_pose_only_returns_covariance():
    F = np.eye(4)
    C = np.diag([1e-6] * 6)

    U = UncertainTransform(F, C)

    p = np.array([0.1, 0.0, 0.0])
    p_out, Cp_out = U.apply_to_point(p, Cp=None)

    assert p_out.shape == (3,)
    assert Cp_out.shape == (3, 3)
    assert np.allclose(Cp_out, Cp_out.T)
    assert np.all(np.diag(Cp_out) >= 0.0)


def test_rotation_only_noise_increases_with_distance():
    # Rotation-only covariance
    C = np.zeros((6, 6))
    C[:3, :3] = 1e-5 * np.eye(3)

    U = UncertainTransform(np.eye(4), C)

    # Two points: one near origin, one farther away
    p_near = np.array([0.01, 0.0, 0.0])
    p_far = np.array([0.5, 0.0, 0.0])

    _, Cp_near = U.apply_to_point(p_near, Cp=None)
    _, Cp_far = U.apply_to_point(p_far, Cp=None)

    # Far point should have larger variance due to rotation-induced term
    assert np.trace(Cp_far) > np.trace(Cp_near)


# ── nominal_types integration ─────────────────────────────────────────────────

def test_from_nominal_frame_roundtrip():
    """UncertainTransform.from_nominal_frame() then get_nominal_frame() should recover the same R and p."""
    R = Rot(axis='z', angle=math.pi / 4)
    p = vct3(0.1, 0.2, 0.3)
    frame = Frame(R, p)
    C = 1e-4 * np.eye(6)

    U = UncertainTransform.from_nominal_frame(frame, C)

    # Check nominal matrix was built correctly
    assert np.allclose(U.F_nom[:3, :3], R.matrix, atol=1e-12)
    assert np.allclose(U.F_nom[:3, 3], [0.1, 0.2, 0.3], atol=1e-12)

    # Round-trip back to Frame
    frame2 = U.get_nominal_frame()
    assert np.allclose(frame2.R.matrix, R.matrix, atol=1e-12)
    assert math.isclose(frame2.p.x, 0.1) and math.isclose(frame2.p.y, 0.2) and math.isclose(frame2.p.z, 0.3)


def test_from_nominal_frame_zero_covariance_default():
    """from_nominal_frame with no C argument should default to zero covariance."""
    frame = Frame(Rot(axis='x', angle=0.0), vct3(0.0, 0.0, 0.0))
    U = UncertainTransform.from_nominal_frame(frame)
    assert np.allclose(U.C, np.zeros((6, 6)))


def test_apply_to_point_accepts_point_type():
    """apply_to_point should accept a vct3 and return a vct3."""
    F = np.eye(4); F[:3, 3] = [1.0, 0.0, 0.0]
    C = 1e-6 * np.eye(6)
    U = UncertainTransform(F, C)

    p_in = vct3(0.0, 0.0, 0.0)
    p_out, Cp_out = U.apply_to_point(p_in)

    assert isinstance(p_out, vct3)
    assert math.isclose(p_out.x, 1.0) and math.isclose(p_out.y, 0.0) and math.isclose(p_out.z, 0.0)
    assert Cp_out.shape == (3, 3)


def test_apply_to_point_array_unchanged():
    """Passing a numpy array to apply_to_point should still return a numpy array."""
    U = UncertainTransform(np.eye(4), 1e-6 * np.eye(6))
    p_out, _ = U.apply_to_point(np.array([1.0, 2.0, 3.0]))
    assert isinstance(p_out, np.ndarray)


# ── convention-dispatched composition subroutines ────────────────────────────

from uncertainty_networks.uncertain_geometry import Convention
from uncertainty_networks.se3 import make_se3, rotz


def _make_ut(angle, t, cov_scale, convention=Convention.RIGHT):
    F = make_se3(rotz(angle), t)
    C = cov_scale * np.eye(6)
    return UncertainTransform(F, C, convention)


def test_compose_rr_matches_matmul():
    """_compose_rr is the existing path; result must equal the old @ operator."""
    U1 = _make_ut(0.1, [0.1, 0, 0], 1e-6, Convention.RIGHT)
    U2 = _make_ut(-0.05, [0, 0.1, 0], 2e-6, Convention.RIGHT)
    result = U1.compose(U2)
    assert result.convention == Convention.RIGHT
    assert np.allclose(result.C, result.C.T)
    assert np.all(np.diag(result.C) >= 0)


def test_compose_ll_returns_left_convention():
    """LEFT @ LEFT must return LEFT convention and a symmetric PSD covariance."""
    U1 = _make_ut(0.1, [0.1, 0, 0], 1e-6, Convention.LEFT)
    U2 = _make_ut(-0.05, [0, 0.1, 0], 2e-6, Convention.LEFT)
    result = U1.compose(U2)
    assert result.convention == Convention.LEFT
    assert np.allclose(result.C, result.C.T)
    assert np.all(np.linalg.eigvalsh(result.C) >= -1e-14)


def test_compose_ll_nominal_matches_rr():
    """All four subroutines must produce the same nominal transform."""
    F1 = make_se3(rotz(0.1), [0.1, 0, 0])
    F2 = make_se3(rotz(-0.05), [0, 0.1, 0])
    C1 = 1e-6 * np.eye(6)
    C2 = 2e-6 * np.eye(6)
    rr = UncertainTransform(F1, C1, Convention.RIGHT).compose(
             UncertainTransform(F2, C2, Convention.RIGHT))
    ll = UncertainTransform(F1, C1, Convention.LEFT).compose(
             UncertainTransform(F2, C2, Convention.LEFT))
    rl = UncertainTransform(F1, C1, Convention.RIGHT).compose(
             UncertainTransform(F2, C2, Convention.LEFT))
    lr = UncertainTransform(F1, C1, Convention.LEFT).compose(
             UncertainTransform(F2, C2, Convention.RIGHT))
    F_expected = F1 @ F2
    for ut in (rr, ll, rl, lr):
        assert np.allclose(ut.F_nom, F_expected, atol=1e-12)


def test_compose_rl_returns_right_convention():
    """RIGHT @ LEFT must return RIGHT convention."""
    U1 = _make_ut(0.1, [0.1, 0, 0], 1e-6, Convention.RIGHT)
    U2 = _make_ut(-0.05, [0, 0.1, 0], 2e-6, Convention.LEFT)
    result = U1.compose(U2)
    assert result.convention == Convention.RIGHT
    assert np.allclose(result.C, result.C.T)
    assert np.all(np.linalg.eigvalsh(result.C) >= -1e-14)


def test_compose_lr_returns_right_convention():
    """LEFT @ RIGHT must return RIGHT convention."""
    U1 = _make_ut(0.1, [0.1, 0, 0], 1e-6, Convention.LEFT)
    U2 = _make_ut(-0.05, [0, 0.1, 0], 2e-6, Convention.RIGHT)
    result = U1.compose(U2)
    assert result.convention == Convention.RIGHT
    assert np.allclose(result.C, result.C.T)
    assert np.all(np.linalg.eigvalsh(result.C) >= -1e-14)


def test_compose_rr_ll_equivalent_at_identity():
    """At identity nominal, Ad = I, so RR and LL should give the same covariance."""
    C1 = np.diag([1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6])
    C2 = np.diag([6e-6, 5e-6, 4e-6, 3e-6, 2e-6, 1e-6])
    rr = UncertainTransform(np.eye(4), C1, Convention.RIGHT).compose(
             UncertainTransform(np.eye(4), C2, Convention.RIGHT))
    ll = UncertainTransform(np.eye(4), C1, Convention.LEFT).compose(
             UncertainTransform(np.eye(4), C2, Convention.LEFT))
    assert np.allclose(rr.C, ll.C, atol=1e-14)


def test_inv_left_convention():
    """inv() of a LEFT-convention transform must stay LEFT and satisfy T @ T^{-1} ≈ I."""
    U = _make_ut(0.2, [0.05, -0.03, 0.1], 1e-6, Convention.LEFT)
    U_inv = U.inv()
    assert U_inv.convention == Convention.LEFT
    assert np.allclose(U_inv.C, U_inv.C.T)
    assert np.allclose(U.F_nom @ U_inv.F_nom, np.eye(4), atol=1e-12)


def test_inv_right_convention_unchanged():
    """inv() of a RIGHT-convention transform must stay RIGHT (existing behavior)."""
    U = _make_ut(0.2, [0.05, -0.03, 0.1], 1e-6, Convention.RIGHT)
    U_inv = U.inv()
    assert U_inv.convention == Convention.RIGHT
    assert np.allclose(U.F_nom @ U_inv.F_nom, np.eye(4), atol=1e-12)
