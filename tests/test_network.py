import numpy as np

from uncertainty_networks import UncertainTransform, GeometricNetwork
from uncertainty_networks import uvct3


def test_find_path_and_query_composition_identity_edges():
    F = np.eye(4)
    C = np.diag([1e-6] * 6)
    U = UncertainTransform(F, C)

    net = GeometricNetwork()
    net.add_edge("A", "B", U, add_inverse=False)
    net.add_edge("B", "C", U, add_inverse=False)

    path = net.find_path("A", "C")
    assert path == ["A", "B", "C"]

    res = net.query("A", "C")
    assert res.path == ["A", "B", "C"]

    # Nominal should remain identity for identity edges
    assert np.allclose(res.transform.F_nom, np.eye(4))

    # For identity F_ab, Ad = I, so covariance should add approximately
    # C_ac = C_ab + C_bc = 2C (since both edges identical)
    assert np.allclose(np.diag(res.transform.C), 2.0 * np.diag(C))


def test_add_inverse_creates_reverse_edge():
    F = np.eye(4)
    C = np.diag([1e-6] * 6)
    U = UncertainTransform(F, C)

    net = GeometricNetwork()
    net.add_edge("A", "B", U, add_inverse=True)

    # Should have B->A edge
    assert net.get_edge("B", "A") is not None

    # Query B->A should exist
    res = net.query("B", "A")
    assert res.path == ["B", "A"]
    assert res.transform.C.shape == (6, 6)


def test_query_raises_when_no_path():
    net = GeometricNetwork()
    net.add_frame("A")
    net.add_frame("B")

    try:
        net.query("A", "B")
        assert False, "Expected ValueError when no path exists"
    except ValueError:
        assert True

def test_query_point_returns_point_and_covariance_in_target_frame():
    import numpy as np
    from uncertainty_networks.se3 import make_se3, rotz

    net = GeometricNetwork()

    # A -> B uncertain transform
    F_ab = make_se3(rotz(0.1), [0.1, 0.0, 0.0])
    C_ab = np.diag([1e-6] * 6)
    U_ab = UncertainTransform(F_ab, C_ab)

    net.add_edge("A", "B", U_ab, add_inverse=True)

    # Point attached to A
    pA = np.array([0.1, 0.0, 0.0])
    CpA = 1e-6 * np.eye(3)
    net.add_point("p1", "A", uvct3(pA, CpA))

    # Query point in B
    up = net.query_point("p1", "B")
    pB  = np.array([up.x, up.y, up.z])
    CpB = up.C

    assert pB.shape == (3,)
    assert CpB.shape == (3, 3)
    assert np.allclose(CpB, CpB.T)
    assert np.all(np.diag(CpB) >= 0.0)

def test_query_relative_vector_and_distance_runs():
    import numpy as np
    from uncertainty_networks.se3 import make_se3, rotz

    net = GeometricNetwork()

    # frames
    net.add_frame("A")
    net.add_frame("B")

    # edge A->B
    U_ab = UncertainTransform(make_se3(rotz(0.05), [0.1, 0.0, 0.0]), np.diag([1e-6]*6))
    net.add_edge("A", "B", U_ab, add_inverse=True)

    # points
    net.add_point("pA", "A", uvct3([0.1, 0.0, 0.0], 1e-6*np.eye(3)))
    net.add_point("pB", "B", uvct3([0.0, 0.1, 0.0], 1e-6*np.eye(3)))

    result  = net.query_relative_vector("pA", "pB", "A")
    delta   = np.array([result.x, result.y, result.z])
    C_delta = result.C
    assert delta.shape == (3,)
    assert C_delta.shape == (3, 3)
    assert np.allclose(C_delta, C_delta.T)

    d, var_d = net.query_distance("pA", "pB", "A")
    assert d >= 0.0
    assert var_d >= 0.0
 
def test_correlation_aware_relative_vector_reduces_cov_when_edges_shared():
    import numpy as np
    from uncertainty_networks import UncertainTransform, GeometricNetwork
    from uncertainty_networks.se3 import make_se3, rotz

    net = GeometricNetwork()

    # Shared edge A <-> B (inverse shares SAME edge_id internally)
    U_ab = UncertainTransform(
        make_se3(rotz(0.1), [0.2, 0.0, 0.0]),
        np.diag([5e-6] * 6),
    )
    net.add_edge("A", "B", U_ab, add_inverse=True)

    # Two points attached to A; query both into B => both depend on SAME edge
    net.add_point("p1", "A", uvct3([0.1, 0.0, 0.0], 1e-6 * np.eye(3)))
    net.add_point("p2", "A", uvct3([0.2, 0.0, 0.0], 1e-6 * np.eye(3)))

    result_ind  = net.query_relative_vector_independent("p1", "p2", "B")
    result_corr = net.query_relative_vector("p1", "p2", "B")
    delta_ind  = np.array([result_ind.x,  result_ind.y,  result_ind.z ])
    delta_corr = np.array([result_corr.x, result_corr.y, result_corr.z])
    C_ind  = result_ind.C
    C_corr = result_corr.C

    # Same mean
    assert np.allclose(delta_ind, delta_corr)

    # Correlation-aware covariance should not be larger than independent baseline (trace-wise)
    assert np.trace(C_corr) <= np.trace(C_ind) + 1e-12
