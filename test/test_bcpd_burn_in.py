"""Tests for BCPD burn-in functionality."""

import math

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from cpdx.bayes import align, align_with_ic
from cpdx.bayes.kernel import gaussian_kernel
from cpdx.bayes.util import initialize


class TestBurnInBaseline:
    """burn_in=0 should match existing behavior."""

    @pytest.fixture
    def points(self):
        key = jax.random.PRNGKey(42)
        ref = jax.random.uniform(key, (30, 3), minval=-5, maxval=5)
        mov = jax.random.uniform(key + 1, (30, 3), minval=-5, maxval=5)
        return ref, mov

    def test_burn_in_zero_matches_baseline_tolerance_none(self, points):
        """burn_in=0 with tolerance=None should match baseline."""
        ref, mov = points
        (P_base, R_base, s_base, t_base, v_base), var_base = align(
            ref, mov, 0.05, 10, None, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
        )
        (P_bi, R_bi, s_bi, t_bi, v_bi), var_bi = align(
            ref, mov, 0.05, 10, None, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
            burn_in=0,
        )
        npt.assert_allclose(R_base, R_bi, atol=1e-6)
        npt.assert_allclose(s_base, s_bi, atol=1e-6)
        npt.assert_allclose(t_base, t_bi, atol=1e-6)
        npt.assert_allclose(v_base, v_bi, atol=1e-6)
        npt.assert_allclose(var_base, var_bi, atol=1e-6)

    def test_burn_in_zero_matches_baseline_with_tolerance(self, points):
        """burn_in=0 with tolerance set should match baseline."""
        ref, mov = points
        (P_base, R_base, s_base, t_base, v_base), var_base = align(
            ref, mov, 0.05, 100, 1e-8, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
        )
        (P_bi, R_bi, s_bi, t_bi, v_bi), var_bi = align(
            ref, mov, 0.05, 100, 1e-8, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
            burn_in=0,
        )
        npt.assert_allclose(R_base, R_bi, atol=1e-6)
        npt.assert_allclose(s_base, s_bi, atol=1e-6)
        npt.assert_allclose(t_base, t_bi, atol=1e-6)
        npt.assert_allclose(v_base, v_bi, atol=1e-6)


class TestBurnInProducesDifferentResult:
    """burn_in > 0 should produce different results than burn_in=0."""

    @pytest.fixture
    def misaligned_points(self):
        """Generate points with moderate misalignment."""
        key = jax.random.PRNGKey(123)
        base = jax.random.uniform(key, (40, 3), minval=-3, maxval=3)
        # Add rotation, scale, translation, and noise
        R = jnp.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        s = 1.3
        t = jnp.array([2.0, -1.0, 0.5])
        mov = s * (base @ R.T) + t[None, :] + jax.random.normal(key + 1, (40, 3)) * 0.1
        ref = base + jax.random.normal(key + 2, (40, 3)) * 0.1
        return ref, mov

    def test_burn_in_changes_result(self, misaligned_points):
        """burn_in=5 should produce different v than burn_in=0."""
        ref, mov = misaligned_points
        (_, _, _, _, v_no), _ = align(
            ref, mov, 0.05, 30, None, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
            burn_in=0,
        )
        (_, _, _, _, v_bi), _ = align(
            ref, mov, 0.05, 30, None, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
            burn_in=5,
        )
        assert v_no.shape == v_bi.shape
        assert not jnp.allclose(v_no, v_bi, atol=1e-4)


class TestVarianceHistory:
    """Variance history should have correct length and content."""

    @pytest.fixture
    def points(self):
        key = jax.random.PRNGKey(42)
        ref = jax.random.uniform(key, (20, 3), minval=-5, maxval=5)
        mov = jax.random.uniform(key + 1, (20, 3), minval=-5, maxval=5)
        return ref, mov

    def test_variance_history_length_tolerance_none(self, points):
        """With tolerance=None, varz shape should equal num_iter."""
        ref, mov = points
        num_iter = 20
        burn_in = 5
        (_, _, _, _, _), varz = align(
            ref, mov, 0.05, num_iter, None, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
            burn_in=burn_in,
        )
        assert varz.shape == (num_iter,), f"Expected shape ({num_iter},), got {varz.shape}"

    def test_variance_history_length_baseline(self, points):
        """Without burn_in, varz shape should equal num_iter."""
        ref, mov = points
        num_iter = 20
        (_, _, _, _, _), varz = align(
            ref, mov, 0.05, num_iter, None, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
        )
        assert varz.shape == (num_iter,)

    def test_total_iterations_with_tolerance(self, points):
        """With tolerance set, iteration count should include burn_in."""
        ref, mov = points
        num_iter = 50
        burn_in = 5
        (_, _, _, _, _), (var_final, total_iters) = align(
            ref, mov, 0.05, num_iter, 1e-8, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
            burn_in=burn_in,
        )
        assert total_iters >= burn_in, f"Expected at least {burn_in} iters, got {total_iters}"
        assert total_iters <= num_iter, f"Expected at most {num_iter} iters, got {total_iters}"


class TestAlignWithICBurnIn:
    """burn_in should work with align_with_ic."""

    @pytest.fixture
    def setup(self):
        key = jax.random.PRNGKey(42)
        ref = jax.random.uniform(key, (25, 3), minval=-5, maxval=5)
        mov = jax.random.uniform(key + 1, (25, 3), minval=-5, maxval=5)
        G, alpha_m, sigma_m, var_i = initialize(ref, mov, gaussian_kernel, 0.5, 1.0)
        R = jnp.eye(3)
        s = jnp.array(1.0)
        t = jnp.zeros(3)
        v = jnp.zeros_like(mov)
        return ref, mov, G, alpha_m, sigma_m, var_i, R, s, t, v

    def test_align_with_ic_burn_in(self, setup):
        """align_with_ic with burn_in should produce valid results."""
        ref, mov, G, alpha_m, sigma_m, var_i, R, s, t, v = setup
        num_iter = 20
        burn_in = 5

        (P, R_out, s_out, t_out, v_out), varz = align_with_ic(
            ref, mov, 0.05, num_iter, None, 1.0, math.inf,
            G, R, s, t, v, sigma_m, alpha_m, var_i,
            burn_in=burn_in,
        )
        assert varz.shape == (num_iter,)
        assert jnp.all(jnp.isfinite(varz))
        assert v_out.shape == mov.shape

    def test_align_with_ic_burn_in_zero_matches(self, setup):
        """align_with_ic with burn_in=0 should match baseline."""
        ref, mov, G, alpha_m, sigma_m, var_i, R, s, t, v = setup

        (P_base, R_base, s_base, t_base, v_base), var_base = align_with_ic(
            ref, mov, 0.05, 15, None, 1.0, math.inf,
            G, R, s, t, v, sigma_m, alpha_m, var_i,
        )
        (P_bi, R_bi, s_bi, t_bi, v_bi), var_bi = align_with_ic(
            ref, mov, 0.05, 15, None, 1.0, math.inf,
            G, R, s, t, v, sigma_m, alpha_m, var_i,
            burn_in=0,
        )
        npt.assert_allclose(R_base, R_bi, atol=1e-6)
        npt.assert_allclose(s_base, s_bi, atol=1e-6)
        npt.assert_allclose(t_base, t_bi, atol=1e-6)
        npt.assert_allclose(v_base, v_bi, atol=1e-6)


class TestInputValidation:
    """Input validation for burn_in parameter."""

    @pytest.fixture
    def points(self):
        key = jax.random.PRNGKey(42)
        ref = jax.random.uniform(key, (20, 3), minval=-5, maxval=5)
        mov = jax.random.uniform(key + 1, (20, 3), minval=-5, maxval=5)
        return ref, mov

    def test_burn_in_negative_raises(self, points):
        """Negative burn_in should raise ValueError."""
        ref, mov = points
        with pytest.raises(ValueError, match="burn_in must be >= 0"):
            align(ref, mov, 0.05, 10, None, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
                  burn_in=-1)

    def test_burn_in_ge_num_iter_raises(self, points):
        """burn_in >= num_iter should raise ValueError."""
        ref, mov = points
        with pytest.raises(ValueError, match="must be less than num_iter"):
            align(ref, mov, 0.05, 10, None, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
                  burn_in=10)

    def test_burn_in_equal_num_iter_raises(self, points):
        """burn_in == num_iter should raise ValueError."""
        ref, mov = points
        with pytest.raises(ValueError, match="must be less than num_iter"):
            align(ref, mov, 0.05, 10, None, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
                  burn_in=10)

    def test_burn_in_negative_raises_align_with_ic(self, points):
        """Negative burn_in in align_with_ic should raise ValueError."""
        ref, mov = points
        G, alpha_m, sigma_m, var_i = initialize(ref, mov, gaussian_kernel, 0.5, 1.0)
        with pytest.raises(ValueError, match="burn_in must be >= 0"):
            align_with_ic(
                ref, mov, 0.05, 10, None, 1.0, math.inf,
                G, jnp.eye(3), jnp.array(1.0), jnp.zeros(3), jnp.zeros_like(mov),
                sigma_m, alpha_m, var_i, burn_in=-1,
            )


class TestTransformModeInteraction:
    """burn_in should be ignored for non-'both' transform modes."""

    @pytest.fixture
    def points(self):
        key = jax.random.PRNGKey(42)
        ref = jax.random.uniform(key, (20, 3), minval=-5, maxval=5)
        mov = jax.random.uniform(key + 1, (20, 3), minval=-5, maxval=5)
        return ref, mov

    def test_burn_in_ignored_for_rigid_mode(self, points):
        """burn_in should not affect rigid-only mode."""
        ref, mov = points
        (_, R_no, s_no, t_no, v_no), var_no = align(
            ref, mov, 0.05, 15, None, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
            transform_mode="rigid", burn_in=0,
        )
        (_, R_bi, s_bi, t_bi, v_bi), var_bi = align(
            ref, mov, 0.05, 15, None, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
            transform_mode="rigid", burn_in=5,
        )
        npt.assert_allclose(R_no, R_bi, atol=1e-6)
        npt.assert_allclose(s_no, s_bi, atol=1e-6)
        npt.assert_allclose(t_no, t_bi, atol=1e-6)
        npt.assert_allclose(v_no, v_bi, atol=1e-6)

    def test_burn_in_ignored_for_nonrigid_mode(self, points):
        """burn_in should not affect nonrigid-only mode."""
        ref, mov = points
        (_, _, _, _, v_no), var_no = align(
            ref, mov, 0.05, 15, None, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
            transform_mode="nonrigid", burn_in=0,
        )
        (_, _, _, _, v_bi), var_bi = align(
            ref, mov, 0.05, 15, None, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
            transform_mode="nonrigid", burn_in=5,
        )
        npt.assert_allclose(v_no, v_bi, atol=1e-6)


class TestConvergence:
    """Burn-in should not hurt convergence."""

    def test_burn_in_converges(self):
        """With burn_in, variance should decrease over iterations."""
        key = jax.random.PRNGKey(99)
        ref = jax.random.uniform(key, (30, 3), minval=-3, maxval=3)
        mov = jax.random.uniform(key + 1, (30, 3), minval=-3, maxval=3)

        (_, _, _, _, _), varz = align(
            ref, mov, 0.05, 30, None, gaussian_kernel, 1.0, 0.5, 1.0, math.inf,
            burn_in=5,
        )
        # Variance should generally decrease (not strictly, but trend should be down)
        assert varz[-1] < varz[0], f"Final variance {varz[-1]} should be < initial {varz[0]}"
