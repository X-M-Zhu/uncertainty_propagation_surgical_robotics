import numpy as np
from uncertainty_networks import UncertainTransform


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
