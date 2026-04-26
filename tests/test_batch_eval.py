import numpy as np
from uncertainty_networks.examples import build_shared_infrastructure_network


def test_evaluate_pairs_runs_empty_and_nonempty():
    net = build_shared_infrastructure_network()

    # empty list should return empty dict
    out0 = net.evaluate_pairs([])
    assert out0 == {}

    # a single pair should return one entry with expected shapes
    pairs = [("p_tip", "p_landmark", "W")]
    out1 = net.evaluate_pairs(pairs, compute_distance=True)

    key = ("p_tip", "p_landmark", "W")
    assert key in out1

    entry = out1[key]
    assert entry["delta_ind"].shape == (3,)
    assert entry["C_ind"].shape == (3, 3)
    assert entry["delta_corr"].shape == (3,)
    assert entry["C_corr"].shape == (3, 3)
    assert isinstance(entry["d"], float)
    assert isinstance(entry["var_d"], float)
    assert np.allclose(entry["C_corr"], entry["C_corr"].T)
