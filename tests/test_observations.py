# Author: X.M. Christine Zhu
# Date: 03/28/2026

"""
Tests for the Observation / Factor Abstraction layer (observations.py).

Coverage
--------
1.  LoopObservation: residual, Jacobian shape, noise defaults.
2.  PointObservation.build: Jacobian formula, residual sign.
3.  DistanceObservation.build: residual, unit-vector Jacobian, same-key merge.
4.  condition_on_observations with a single LoopObservation matches
    the existing condition_on_loop result exactly.
5.  Posterior trace strictly decreases as more observations are stacked.
6.  Posterior covariance is symmetric and positive definite.
7.  Mixed observations (loop + point + distance) run without error.
8.  KeyError on unknown state key.
9.  cross_cov helper on ConditioningResult.
"""

import numpy as np
import pytest

from uncertainty_networks.se3 import exp_se3
from uncertainty_networks.closed_loop import (
    linearize_loop_residual,
    condition_on_loop,
)
from uncertainty_networks.observations import (
    LoopObservation,
    PointObservation,
    DistanceObservation,
    condition_on_observations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_cov(d: int, rng: np.random.Generator, scale: float = 1e-3) -> np.ndarray:
    """Generate a random SPD covariance of size d×d."""
    A = rng.standard_normal((d, d))
    return scale * (A @ A.T + np.eye(d))


def _rand_F(rng: np.random.Generator, angle_scale: float = 0.3) -> np.ndarray:
    """Generate a random SE(3) nominal transform."""
    xi = rng.standard_normal(6)
    xi[:3] *= angle_scale
    xi[3:] *= 0.1
    return exp_se3(xi)


# ---------------------------------------------------------------------------
# 1. LoopObservation basics
# ---------------------------------------------------------------------------

class TestLoopObservation:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.F_res = _rand_F(rng)
        self.F_k   = _rand_F(rng)

    def test_residual_shape(self):
        obs = LoopObservation(self.F_res, self.F_k, "res", "k")
        assert obs.residual().shape == (6,)

    def test_jacobian_shapes(self):
        obs = LoopObservation(self.F_res, self.F_k, "res", "k")
        jacs = obs.jacobians()
        assert set(jacs.keys()) == {"res", "k"}
        assert jacs["res"].shape == (6, 6)
        assert jacs["k"].shape == (6, 6)

    def test_noise_default_is_tight(self):
        obs = LoopObservation(self.F_res, self.F_k, "res", "k")
        C_nu = obs.noise_cov()
        assert C_nu.shape == (6, 6)
        # Default is 1e-9 * I
        np.testing.assert_allclose(C_nu, 1e-9 * np.eye(6))

    def test_noise_custom(self):
        C_nu = 1e-4 * np.eye(6)
        obs = LoopObservation(self.F_res, self.F_k, "res", "k", C_nu=C_nu)
        np.testing.assert_allclose(obs.noise_cov(), C_nu)

    def test_state_keys_property(self):
        obs = LoopObservation(self.F_res, self.F_k, "path_A", "path_B")
        assert set(obs.state_keys) == {"path_A", "path_B"}

    def test_dim(self):
        obs = LoopObservation(self.F_res, self.F_k, "res", "k")
        assert obs.dim == 6

    def test_jacobians_match_linearize_loop_residual(self):
        obs = LoopObservation(self.F_res, self.F_k, "res", "k")
        lin = linearize_loop_residual(self.F_res, self.F_k)
        np.testing.assert_allclose(obs.residual(), lin.r0, atol=1e-12)
        np.testing.assert_allclose(obs.jacobians()["res"], lin.J_res, atol=1e-12)
        np.testing.assert_allclose(obs.jacobians()["k"],   lin.J_k,   atol=1e-12)


# ---------------------------------------------------------------------------
# 2. PointObservation basics
# ---------------------------------------------------------------------------

class TestPointObservation:
    def setup_method(self):
        self.p_nom = np.array([0.1, 0.2, 0.3])
        self.z     = np.array([0.09, 0.21, 0.31])
        self.C_nu  = 1e-4 * np.eye(3)

    def test_residual(self):
        obs = PointObservation.build(self.p_nom, "path", self.z, self.C_nu)
        np.testing.assert_allclose(obs.residual(), self.p_nom - self.z)

    def test_jacobian_shape(self):
        obs = PointObservation.build(self.p_nom, "path", self.z, self.C_nu)
        J = obs.jacobians()["path"]
        assert J.shape == (3, 6)

    def test_jacobian_cis_i_formula(self):
        """J_eta = [-[p_nom]x | I_3] for CIS I left perturbation."""
        from uncertainty_networks.se3 import skew
        obs = PointObservation.build(self.p_nom, "path", self.z, self.C_nu)
        J = obs.jacobians()["path"]
        J_expected = np.hstack([-skew(self.p_nom), np.eye(3)])
        np.testing.assert_allclose(J, J_expected, atol=1e-15)

    def test_dim(self):
        obs = PointObservation.build(self.p_nom, "path", self.z, self.C_nu)
        assert obs.dim == 3

    def test_noise_stored(self):
        obs = PointObservation.build(self.p_nom, "path", self.z, self.C_nu)
        np.testing.assert_allclose(obs.noise_cov(), self.C_nu)

    def test_zero_residual_when_z_equals_nom(self):
        obs = PointObservation.build(self.p_nom, "path", self.p_nom, self.C_nu)
        np.testing.assert_allclose(obs.residual(), np.zeros(3), atol=1e-15)


# ---------------------------------------------------------------------------
# 3. DistanceObservation basics
# ---------------------------------------------------------------------------

class TestDistanceObservation:
    def setup_method(self):
        self.p1 = np.array([1.0, 0.0, 0.0])
        self.p2 = np.array([0.0, 0.0, 0.0])
        self.d_nom = 1.0          # ||p1 - p2||
        self.z = 1.05             # slightly different measured distance
        self.sigma = 1e-3

    def test_residual(self):
        obs = DistanceObservation.build(self.p1, self.p2, "k1", "k2", self.z, self.sigma)
        r = obs.residual()
        assert r.shape == (1,)
        np.testing.assert_allclose(r[0], self.d_nom - self.z, atol=1e-12)

    def test_jacobian_shapes(self):
        obs = DistanceObservation.build(self.p1, self.p2, "k1", "k2", self.z, self.sigma)
        jacs = obs.jacobians()
        assert set(jacs.keys()) == {"k1", "k2"}
        assert jacs["k1"].shape == (1, 6)
        assert jacs["k2"].shape == (1, 6)

    def test_noise_cov(self):
        obs = DistanceObservation.build(self.p1, self.p2, "k1", "k2", self.z, self.sigma)
        C_nu = obs.noise_cov()
        assert C_nu.shape == (1, 1)
        np.testing.assert_allclose(C_nu[0, 0], self.sigma ** 2)

    def test_unit_vector_direction(self):
        """Jacobian for k1 should align with unit vector p1 - p2."""
        obs = DistanceObservation.build(self.p1, self.p2, "k1", "k2", self.z, self.sigma)
        # p1 - p2 = [1, 0, 0], unit = [1, 0, 0]
        # J1 = u^T J_eta_1 = [1,0,0] @ [-skew(p1) | I]
        from uncertainty_networks.se3 import skew
        J_eta_1 = np.hstack([-skew(self.p1), np.eye(3)])
        J1_expected = np.array([[1.0, 0.0, 0.0]]) @ J_eta_1
        np.testing.assert_allclose(obs.jacobians()["k1"], J1_expected.reshape(1, 6), atol=1e-12)

    def test_same_key_merges_jacobians(self):
        """When key_1 == key_2, jacobians() returns a single key with summed J."""
        obs = DistanceObservation.build(self.p1, self.p2, "shared", "shared", self.z, self.sigma)
        jacs = obs.jacobians()
        assert set(jacs.keys()) == {"shared"}
        assert jacs["shared"].shape == (1, 6)

    def test_zero_residual_when_z_equals_nom(self):
        obs = DistanceObservation.build(self.p1, self.p2, "k1", "k2", self.d_nom, self.sigma)
        np.testing.assert_allclose(obs.residual()[0], 0.0, atol=1e-12)

    def test_dim(self):
        obs = DistanceObservation.build(self.p1, self.p2, "k1", "k2", self.z, self.sigma)
        assert obs.dim == 1


# ---------------------------------------------------------------------------
# 4. condition_on_observations matches condition_on_loop (single loop case)
# ---------------------------------------------------------------------------

class TestConditionOnObservationsMatchesLoop:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.F_res = _rand_F(rng)
        self.F_k   = _rand_F(rng)
        self.C_res = _rand_cov(6, rng)
        self.C_k   = _rand_cov(6, rng)
        self.C_nu  = 1e-6 * np.eye(6)

    def test_posterior_matches_condition_on_loop(self):
        lin = linearize_loop_residual(self.F_res, self.F_k)
        ref = condition_on_loop(self.C_res, self.C_k, lin, C_nu=self.C_nu)

        obs = LoopObservation(self.F_res, self.F_k, "res", "k", C_nu=self.C_nu)
        result = condition_on_observations(
            priors={"res": self.C_res, "k": self.C_k},
            observations=[obs],
        )

        np.testing.assert_allclose(result.posteriors["res"], ref.C_res, atol=1e-10)
        np.testing.assert_allclose(result.posteriors["k"],   ref.C_k,   atol=1e-10)

    def test_full_posterior_matches(self):
        lin = linearize_loop_residual(self.F_res, self.F_k)
        ref = condition_on_loop(self.C_res, self.C_k, lin, C_nu=self.C_nu)

        obs = LoopObservation(self.F_res, self.F_k, "res", "k", C_nu=self.C_nu)
        result = condition_on_observations(
            priors={"res": self.C_res, "k": self.C_k},
            observations=[obs],
        )

        # Keys are sorted: "k" first, then "res"
        idx_res = result.keys.index("res")
        idx_k   = result.keys.index("k")
        s_res = 6 * idx_res
        s_k   = 6 * idx_k

        np.testing.assert_allclose(
            result.C_full[s_res:s_res+6, s_res:s_res+6], ref.C_res, atol=1e-10
        )
        np.testing.assert_allclose(
            result.C_full[s_k:s_k+6, s_k:s_k+6], ref.C_k, atol=1e-10
        )


# ---------------------------------------------------------------------------
# 5. Trace decreases as observations are stacked
# ---------------------------------------------------------------------------

class TestTraceDecreasesWithMoreObservations:
    def setup_method(self):
        rng = np.random.default_rng(7)
        self.C_res = _rand_cov(6, rng)
        # Three alternative paths sharing the same residual
        self.paths = [(_rand_F(rng), _rand_cov(6, rng)) for _ in range(3)]

    def test_trace_decreases_loop_observations(self):
        traces = []
        priors = {"res": self.C_res}
        observations = []

        for i, (F_k, C_k) in enumerate(self.paths):
            F_res = exp_se3(np.zeros(6))   # identity nominal for simplicity
            key_k = f"k{i}"
            priors[key_k] = C_k
            observations.append(
                LoopObservation(F_res, F_k, "res", key_k)
            )
            result = condition_on_observations(priors=priors, observations=observations)
            traces.append(np.trace(result.posteriors["res"]))

        for i in range(len(traces) - 1):
            assert traces[i + 1] < traces[i], (
                f"Trace did not decrease: trace[{i}]={traces[i]:.6g}, "
                f"trace[{i+1}]={traces[i+1]:.6g}"
            )


# ---------------------------------------------------------------------------
# 6. Posterior is symmetric and positive definite
# ---------------------------------------------------------------------------

class TestPosteriorSymmetricPD:
    def setup_method(self):
        rng = np.random.default_rng(99)
        self.F_res = _rand_F(rng)
        self.F_k   = _rand_F(rng)
        self.C_res = _rand_cov(6, rng)
        self.C_k   = _rand_cov(6, rng)

    def _check_spd(self, C: np.ndarray, name: str):
        np.testing.assert_allclose(C, C.T, atol=1e-10, err_msg=f"{name} not symmetric")
        eigvals = np.linalg.eigvalsh(C)
        assert eigvals.min() > 0, f"{name} not positive definite, min eigenvalue={eigvals.min()}"

    def test_posteriors_are_spd(self):
        obs = LoopObservation(self.F_res, self.F_k, "res", "k")
        result = condition_on_observations(
            priors={"res": self.C_res, "k": self.C_k},
            observations=[obs],
        )
        self._check_spd(result.posteriors["res"], "C_res_post")
        self._check_spd(result.posteriors["k"], "C_k_post")

    def test_c_full_is_spd(self):
        obs = LoopObservation(self.F_res, self.F_k, "res", "k")
        result = condition_on_observations(
            priors={"res": self.C_res, "k": self.C_k},
            observations=[obs],
        )
        self._check_spd(result.C_full, "C_full")

    def test_posterior_smaller_than_prior(self):
        """Each diagonal block of the posterior should have smaller trace than prior."""
        obs = LoopObservation(self.F_res, self.F_k, "res", "k")
        result = condition_on_observations(
            priors={"res": self.C_res, "k": self.C_k},
            observations=[obs],
        )
        assert np.trace(result.posteriors["res"]) < np.trace(self.C_res)
        assert np.trace(result.posteriors["k"])   < np.trace(self.C_k)


# ---------------------------------------------------------------------------
# 7. Mixed observation types run without error
# ---------------------------------------------------------------------------

class TestMixedObservations:
    def test_loop_plus_point_plus_distance(self):
        rng = np.random.default_rng(55)
        F_res = _rand_F(rng)
        F_k   = _rand_F(rng)
        C_res = _rand_cov(6, rng)
        C_k   = _rand_cov(6, rng)

        p_nom = rng.standard_normal(3) * 0.1
        z_pt  = p_nom + rng.standard_normal(3) * 0.001
        C_pt  = 1e-4 * np.eye(3)

        p1 = rng.standard_normal(3) * 0.1
        p2 = rng.standard_normal(3) * 0.1
        d_nom = float(np.linalg.norm(p1 - p2))
        z_d   = d_nom + 0.001

        obs_loop  = LoopObservation(F_res, F_k, "res", "k")
        obs_point = PointObservation.build(p_nom, "res", z_pt, C_pt)
        obs_dist  = DistanceObservation.build(p1, p2, "res", "k", z_d, sigma=1e-3)

        result = condition_on_observations(
            priors={"res": C_res, "k": C_k},
            observations=[obs_loop, obs_point, obs_dist],
        )

        assert set(result.posteriors.keys()) == {"res", "k"}
        assert result.C_full.shape == (12, 12)
        # All posteriors should be PSD
        for key, C in result.posteriors.items():
            eigvals = np.linalg.eigvalsh(C)
            assert eigvals.min() > 0, f"posterior['{key}'] not PD"

    def test_point_only_observation(self):
        """PointObservation alone (no loop) should also condition correctly."""
        rng = np.random.default_rng(77)
        C_path = _rand_cov(6, rng)
        p_nom  = np.array([0.05, 0.10, 0.15])
        z      = p_nom + np.array([0.001, -0.001, 0.002])
        C_nu   = 1e-5 * np.eye(3)

        obs = PointObservation.build(p_nom, "path", z, C_nu)
        result = condition_on_observations(
            priors={"path": C_path},
            observations=[obs],
        )
        assert "path" in result.posteriors
        assert np.trace(result.posteriors["path"]) < np.trace(C_path)


# ---------------------------------------------------------------------------
# 8. KeyError on unknown state key
# ---------------------------------------------------------------------------

class TestKeyError:
    def test_unknown_state_key_raises(self):
        rng = np.random.default_rng(3)
        F_res = _rand_F(rng)
        F_k   = _rand_F(rng)
        C_res = _rand_cov(6, rng)
        C_k   = _rand_cov(6, rng)

        obs = LoopObservation(F_res, F_k, "res", "UNKNOWN_KEY")
        with pytest.raises(KeyError, match="UNKNOWN_KEY"):
            condition_on_observations(
                priors={"res": C_res, "k": C_k},
                observations=[obs],
            )


# ---------------------------------------------------------------------------
# 9. cross_cov helper
# ---------------------------------------------------------------------------

class TestCrossCov:
    def test_cross_cov_shape_and_symmetry(self):
        rng = np.random.default_rng(21)
        F_res = _rand_F(rng)
        F_k   = _rand_F(rng)
        C_res = _rand_cov(6, rng)
        C_k   = _rand_cov(6, rng)

        obs = LoopObservation(F_res, F_k, "res", "k")
        result = condition_on_observations(
            priors={"res": C_res, "k": C_k},
            observations=[obs],
        )
        cross = result.cross_cov("res", "k")
        assert cross.shape == (6, 6)
        # cross_cov(A, B)^T should equal cross_cov(B, A)
        cross_T = result.cross_cov("k", "res")
        np.testing.assert_allclose(cross, cross_T.T, atol=1e-12)

    def test_cross_cov_nonzero_after_loop(self):
        """Loop constraint should introduce non-zero cross-covariance."""
        rng = np.random.default_rng(22)
        F_res = _rand_F(rng)
        F_k   = _rand_F(rng)
        C_res = _rand_cov(6, rng)
        C_k   = _rand_cov(6, rng)

        obs = LoopObservation(F_res, F_k, "res", "k")
        result = condition_on_observations(
            priors={"res": C_res, "k": C_k},
            observations=[obs],
        )
        cross = result.cross_cov("res", "k")
        assert np.linalg.norm(cross) > 1e-10, "Cross-covariance should be non-zero after loop"
