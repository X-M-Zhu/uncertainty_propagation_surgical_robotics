import numpy as np
from uncertainty_networks import UncertainTransform, GeometricNetwork
from uncertainty_networks import uvct3
from uncertainty_networks.se3 import make_se3, rotz


def test_corr_shared_infrastructure_trace_not_increase():
    net = GeometricNetwork()
    net.add_frame("W")

    # Shared world connections
    net.add_edge("Rb", "W", UncertainTransform(make_se3(rotz(0.01), [-0.02, 0.02, 0.0]), np.diag([6e-6]*6)), add_inverse=True)
    net.add_edge("CT", "W", UncertainTransform(make_se3(rotz(-0.01), [0.03, -0.01, 0.0]), np.diag([5e-6]*6)), add_inverse=True)

    # Branches
    net.add_edge("Rb", "Tool", UncertainTransform(make_se3(rotz(0.02), [0.0, 0.1, 0.02]), np.diag([4e-6]*6)), add_inverse=True)
    net.add_edge("CT", "Anat", UncertainTransform(make_se3(rotz(0.02), [0.0, 0.0, 0.1]), np.diag([3e-6]*6)), add_inverse=True)

    Cp = 2e-6 * np.eye(3)
    net.add_point("p_tip",  "Tool", uvct3([0.0,  0.0,  -0.12], Cp))
    net.add_point("p_land", "Anat", uvct3([0.03, -0.01, 0.02], Cp))

    result_ind  = net.query_relative_vector_independent("p_tip", "p_land", "W")
    result_corr = net.query_relative_vector("p_tip", "p_land", "W")
    C_ind  = result_ind.C
    C_corr = result_corr.C

    assert np.allclose(C_ind, C_ind.T)
    assert np.allclose(C_corr, C_corr.T)
    assert np.all(np.diag(C_corr) >= 0.0)
    assert np.trace(C_corr) <= np.trace(C_ind) + 1e-12
