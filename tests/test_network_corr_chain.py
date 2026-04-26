import numpy as np
from uncertainty_networks import UncertainTransform, GeometricNetwork
from uncertainty_networks.se3 import make_se3, rotz


def test_corr_chain_trace_not_increase():
    net = GeometricNetwork()

    net.add_edge("A", "B", UncertainTransform(make_se3(rotz(0.07), [0.1, 0, 0]), np.diag([3e-6]*6)), add_inverse=True)
    net.add_edge("B", "C", UncertainTransform(make_se3(rotz(-0.04), [0, 0.12, 0]), np.diag([4e-6]*6)), add_inverse=True)
    net.add_edge("C", "D", UncertainTransform(make_se3(rotz(0.03), [0, 0, 0.08]), np.diag([5e-6]*6)), add_inverse=True)

    Cp = 2e-6 * np.eye(3)
    net.add_point("p1", "A", [0.05, 0.01, 0.0], Cp)
    net.add_point("p2", "C", [0.02, -0.01, 0.01], Cp)

    _, C_ind = net.query_relative_vector_independent("p1", "p2", "D")
    _, C_corr = net.query_relative_vector("p1", "p2", "D")

    assert np.allclose(C_ind, C_ind.T)
    assert np.allclose(C_corr, C_corr.T)
    assert np.all(np.diag(C_corr) >= 0.0)
    assert np.trace(C_corr) <= np.trace(C_ind) + 1e-12
