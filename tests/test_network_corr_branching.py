import numpy as np
from uncertainty_networks import UncertainTransform, GeometricNetwork
from uncertainty_networks import uvct3
from uncertainty_networks.se3 import make_se3, rotz


def test_corr_branching_matches_independent_when_no_shared_edges():
    net = GeometricNetwork()

    net.add_edge("A", "B", UncertainTransform(make_se3(rotz(0.05), [0.1, 0, 0]), np.diag([3e-6]*6)), add_inverse=True)
    net.add_edge("B", "D", UncertainTransform(make_se3(rotz(0.02), [0, 0.1, 0.02]), np.diag([6e-6]*6)), add_inverse=True)

    net.add_edge("A", "C", UncertainTransform(make_se3(rotz(-0.04), [0.1, 0.02, 0]), np.diag([3e-6]*6)), add_inverse=True)
    net.add_edge("C", "D", UncertainTransform(make_se3(rotz(0.01), [0, 0.08, 0.02]), np.diag([6e-6]*6)), add_inverse=True)

    Cp = 2e-6 * np.eye(3)
    net.add_point("p1", "B", uvct3([ 0.02,  0.0,  0.0], Cp))
    net.add_point("p2", "C", uvct3([-0.01,  0.01, 0.0], Cp))

    result_ind  = net.query_relative_vector_independent("p1", "p2", "D")
    result_corr = net.query_relative_vector("p1", "p2", "D")
    C_ind  = result_ind.C
    C_corr = result_corr.C

    # If there are truly no shared edges, corr-aware should match independent (within numerical tolerance)
    assert np.allclose(C_corr, C_ind, atol=1e-10)
