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


def test_transform_point_pose_only_returns_covariance():
    F = np.eye(4)
    C = np.diag([1e-6] * 6)

    U = UncertainTransform(F, C)

    p = np.array([0.1, 0.0, 0.0])
    p_out, Cp_out = U.transform_point(p, Cp=None)

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

    _, Cp_near = U.transform_point(p_near, Cp=None)
    _, Cp_far = U.transform_point(p_far, Cp=None)

    # Far point should have larger variance due to rotation-induced term
    assert np.trace(Cp_far) > np.trace(Cp_near)


# ── nominal_types integration ─────────────────────────────────────────────────

def test_from_frame_roundtrip():
    """UncertainTransform.from_frame() then to_frame() should recover the same R and p."""
    R = Rot(axis='z', angle=math.pi / 4)
    p = vct3(0.1, 0.2, 0.3)
    frame = Frame(R, p)
    C = 1e-4 * np.eye(6)

    U = UncertainTransform.from_frame(frame, C)

    # Check nominal matrix was built correctly
    assert np.allclose(U.F_nom[:3, :3], R.matrix, atol=1e-12)
    assert np.allclose(U.F_nom[:3, 3], [0.1, 0.2, 0.3], atol=1e-12)

    # Round-trip back to Frame
    frame2 = U.to_frame()
    assert np.allclose(frame2.R.matrix, R.matrix, atol=1e-12)
    assert math.isclose(frame2.p.x, 0.1) and math.isclose(frame2.p.y, 0.2) and math.isclose(frame2.p.z, 0.3)


def test_from_frame_zero_covariance_default():
    """from_frame with no C argument should default to zero covariance."""
    frame = Frame(Rot(axis='x', angle=0.0), vct3(0.0, 0.0, 0.0))
    U = UncertainTransform.from_frame(frame)
    assert np.allclose(U.C, np.zeros((6, 6)))


def test_transform_point_accepts_point_type():
    """transform_point should accept a vct3 and return a vct3."""
    F = np.eye(4); F[:3, 3] = [1.0, 0.0, 0.0]
    C = 1e-6 * np.eye(6)
    U = UncertainTransform(F, C)

    p_in = vct3(0.0, 0.0, 0.0)
    p_out, Cp_out = U.transform_point(p_in)

    assert isinstance(p_out, vct3)
    assert math.isclose(p_out.x, 1.0) and math.isclose(p_out.y, 0.0) and math.isclose(p_out.z, 0.0)
    assert Cp_out.shape == (3, 3)


def test_transform_point_array_unchanged():
    """Passing a numpy array to transform_point should still return a numpy array."""
    U = UncertainTransform(np.eye(4), 1e-6 * np.eye(6))
    p_out, _ = U.transform_point(np.array([1.0, 2.0, 3.0]))
    assert isinstance(p_out, np.ndarray)
