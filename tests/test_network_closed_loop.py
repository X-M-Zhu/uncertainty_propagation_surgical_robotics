import numpy as np
from uncertainty_networks import GeometricNetwork, UncertainTransform
from uncertainty_networks.se3 import make_se3, rotz


def test_network_closed_loop_posterior_runs_and_reduces_trace():
    net = GeometricNetwork()

    # Build a small loop: A -> B -> D and A -> C -> D
    C1 = np.diag([3e-6] * 6)
    C2 = np.diag([4e-6] * 6)
    C3 = np.diag([3e-6] * 6)
    C4 = np.diag([4e-6] * 6)

    net.add_edge("A", "B", UncertainTransform(make_se3(rotz(0.05), [0.1, 0.0, 0.0]), C1), add_inverse=True)
    net.add_edge("B", "D", UncertainTransform(make_se3(rotz(0.02), [0.0, 0.1, 0.02]), C2), add_inverse=True)

    net.add_edge("A", "C", UncertainTransform(make_se3(rotz(-0.04), [0.1, 0.02, 0.0]), C3), add_inverse=True)
    net.add_edge("C", "D", UncertainTransform(make_se3(rotz(0.01), [0.0, 0.08, 0.02]), C4), add_inverse=True)

    # Two different paths from A to D
    path_res = ["A", "B", "D"]
    path_k = ["A", "C", "D"]

    # Prior covariances for each path transform
    U_res = net.query_transform_on_path(path_res).transform
    U_k = net.query_transform_on_path(path_k).transform

    post = net.query_closed_loop_posterior(path_res, path_k, C_nu=1e-3 * np.eye(6))

    # Symmetry
    assert np.allclose(post.C_res, post.C_res.T)
    assert np.allclose(post.C_k, post.C_k.T)

    # Conditioning should not increase trace
    assert np.trace(post.C_res) <= np.trace(U_res.C) + 1e-12
    assert np.trace(post.C_k) <= np.trace(U_k.C) + 1e-12
