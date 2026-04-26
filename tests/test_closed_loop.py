import numpy as np

from uncertainty_networks.se3 import make_se3, rotz
from uncertainty_networks.closed_loop import (
    linearize_loop_residual,
    condition_on_loop,
    condition_on_loop_subspace,
)


def test_closed_loop_conditioning_reduces_trace_tight_constraint():
    # Two nominal transforms
    F_res = make_se3(rotz(0.15), [0.10, 0.02, -0.01])
    F_k = make_se3(rotz(0.10), [0.08, -0.01, 0.03])

    # Prior covariances
    C_res = np.diag([6e-6, 6e-6, 6e-6, 8e-6, 8e-6, 8e-6])
    C_k = np.diag([5e-6, 5e-6, 5e-6, 7e-6, 7e-6, 7e-6])

    lin = linearize_loop_residual(F_res, F_k)

    post = condition_on_loop(C_res, C_k, lin, C_nu=1e-9 * np.eye(6))

    # Posterior should be symmetric
    assert np.allclose(post.C_res, post.C_res.T)
    assert np.allclose(post.C_k, post.C_k.T)

    # Tight constraint should reduce uncertainty (trace-wise)
    assert np.trace(post.C_res) <= np.trace(C_res) + 1e-12
    assert np.trace(post.C_k) <= np.trace(C_k) + 1e-12


def test_closed_loop_subspace_alpha_only_runs_and_reduces_alpha_trace():
    F_res = make_se3(rotz(0.05), [0.02, 0.00, 0.01])
    F_k = make_se3(rotz(-0.03), [0.00, 0.03, 0.02])

    C_res = np.diag([6e-6, 6e-6, 6e-6, 8e-6, 8e-6, 8e-6])
    C_k = np.diag([5e-6, 5e-6, 5e-6, 7e-6, 7e-6, 7e-6])

    lin = linearize_loop_residual(F_res, F_k)

    # alpha-only indices = [0,1,2]
    post = condition_on_loop_subspace(C_res, C_k, lin, indices=[0, 1, 2], C_nu_sub=1e-9 * np.eye(3))

    # alpha trace should drop (trace of top-left 3x3)
    assert np.trace(post.C_res[:3, :3]) <= np.trace(C_res[:3, :3]) + 1e-12
    assert np.trace(post.C_k[:3, :3]) <= np.trace(C_k[:3, :3]) + 1e-12
