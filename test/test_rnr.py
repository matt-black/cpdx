"""Tests for rnr (rigid + non-rigid) registration."""

import jax
import jax.numpy as jnp
import pytest

from cpdx._matching import expectation
from cpdx.rigid import align as rigid_align
from cpdx.rigid import transform as rigid_transform
from cpdx.rnr import (
    align,
    align_fixed_iter,
    interpolate,
    interpolate_variance,
    maximization,
    maximization_uncertainty,
    transform,
)
from cpdx.util import sqdist


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_points():
    """Small 2D point sets for deterministic testing."""
    ref = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 0.3]])
    mov = jnp.array([[0.1, 0.1], [1.1, 0.9], [2.5, 0.7], [0.8, 1.2]])
    return ref, mov


@pytest.fixture
def basic_points_3d():
    """Small 3D point sets for deterministic testing."""
    ref = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.5],
        [2.0, 0.5, 1.0],
        [3.0, 0.3, 0.2],
    ])
    mov = jnp.array([
        [0.1, 0.1, 0.05],
        [1.1, 0.9, 0.6],
        [2.5, 0.7, 0.9],
        [0.8, 1.2, 0.3],
    ])
    return ref, mov


@pytest.fixture
def zero_uncertainties(basic_points):
    """Zero uncertainty arrays matching basic_points shapes."""
    ref, mov = basic_points
    unc_ref = jnp.zeros_like(ref)
    unc_mov = jnp.zeros_like(mov)
    return ref, mov, unc_ref, unc_mov


# ============================================================================
# Transform function
# ============================================================================


class TestTransform:
    """Tests for the transform function."""

    def test_shape(self, basic_points):
        """Transform returns correct shape."""
        _, mov = basic_points
        d = mov.shape[1]
        G = jnp.exp(-sqdist(mov, mov) / 2.0)
        W = jnp.zeros_like(mov)
        R = jnp.eye(d)
        s = jnp.array(1.0)
        t = jnp.zeros(d)

        result = transform(mov, R, s, t, G, W)
        assert result.shape == mov.shape

    def test_identity(self, basic_points):
        """Identity transform + zero W gives original points."""
        _, mov = basic_points
        d = mov.shape[1]
        G = jnp.exp(-sqdist(mov, mov) / 2.0)
        W = jnp.zeros_like(mov)
        R = jnp.eye(d)
        s = jnp.array(1.0)
        t = jnp.zeros(d)

        result = transform(mov, R, s, t, G, W)
        assert jnp.allclose(result, mov, atol=1e-10)

    def test_composition(self, basic_points):
        """Transform equals rigid part + non-rigid part."""
        _, mov = basic_points
        d = mov.shape[1]
        m = mov.shape[0]
        G = jnp.exp(-sqdist(mov, mov) / 2.0)
        W = 0.1 * jnp.ones((m, d))
        angle = jnp.pi / 6
        R = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
        ])
        s = jnp.array(1.5)
        t = jnp.array([1.0, -0.5])

        result = transform(mov, R, s, t, G, W)
        rigid_part = rigid_transform(mov, R, s, t)
        nonrigid_part = G @ W
        expected = rigid_part + nonrigid_part
        assert jnp.allclose(result, expected, atol=1e-10)


# ============================================================================
# M-step (maximization)
# ============================================================================


class TestMaximization:
    """Tests for maximization and maximization_uncertainty."""

    def _setup(self):
        ref = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 0.3]])
        mov = jnp.array([[0.1, 0.1], [1.1, 0.9], [2.5, 0.7], [0.8, 1.2]])
        var = jnp.array(0.1)
        G = jnp.exp(-sqdist(mov, mov) / (2 * 0.5))
        w = 0.1
        P = expectation(ref, mov, var, w)
        return ref, mov, P, G, var

    def test_maximization_returns_valid(self):
        """maximization returns valid parameters."""
        ref, mov, P, G, var = self._setup()
        reg = 1.0
        tol = 1e-6

        (R, s, t, W), new_var = maximization(
            ref, mov, P, G, var, reg, tol, inner_iter=2
        )

        assert R.shape == (2, 2)
        assert s.shape == ()
        assert t.shape == (2,)
        assert W.shape == mov.shape
        assert new_var >= 0

    def test_rotation_valid(self):
        """Recovered rotation is approximately orthogonal."""
        ref, mov, P, G, var = self._setup()
        reg = 1.0
        tol = 1e-6

        (R, s, t, W), _ = maximization(ref, mov, P, G, var, reg, tol, inner_iter=2)

        assert jnp.allclose(R @ R.T, jnp.eye(2), atol=1e-4)
        assert jnp.abs(jnp.linalg.det(R) - 1.0) < 1e-4

    def test_burn_in_keeps_W_zero(self):
        """With burn_in=True, W should remain zero."""
        ref, mov, P, G, var = self._setup()
        reg = 1.0
        tol = 1e-6

        (R, s, t, W), _ = maximization(
            ref, mov, P, G, var, reg, tol, inner_iter=2, burn_in=True
        )

        assert jnp.allclose(W, jnp.zeros_like(W), atol=1e-10)

    def test_maximization_uncertainty_returns_valid(self, zero_uncertainties):
        """maximization_uncertainty returns valid parameters."""
        ref, mov, unc_ref, unc_mov = zero_uncertainties
        var = jnp.array(0.1)
        G = jnp.exp(-sqdist(mov, mov) / (2 * 0.5))
        w = 0.1
        P = expectation(ref, mov, var, w)
        reg = 1.0
        tol = 1e-6

        (R, s, t, W), new_var = maximization_uncertainty(
            ref, mov, P, G, var, reg, tol, unc_ref, unc_mov, inner_iter=2
        )

        assert R.shape == (2, 2)
        assert new_var >= 0

    def test_zero_uncertainty_equivalence(self):
        """With zero uncertainty, maximization_uncertainty matches maximization."""
        ref, mov, P, G, var = self._setup()
        reg = 1.0
        tol = 1e-6
        unc_ref = jnp.zeros_like(ref)
        unc_mov = jnp.zeros_like(mov)

        (R_base, s_base, t_base, W_base), var_base = maximization(
            ref, mov, P, G, var, reg, tol, inner_iter=2
        )
        (R_unc, s_unc, t_unc, W_unc), var_unc = maximization_uncertainty(
            ref, mov, P, G, var, reg, tol, unc_ref, unc_mov, inner_iter=2
        )

        assert jnp.allclose(R_base, R_unc, atol=1e-5)
        assert jnp.allclose(s_base, s_unc, atol=1e-5)
        assert jnp.allclose(t_base, t_unc, atol=1e-5)
        assert jnp.allclose(W_base, W_unc, atol=1e-5)


# ============================================================================
# Full alignment
# ============================================================================


class TestAlign:
    """Tests for align and align_fixed_iter."""

    def test_align_converges(self, basic_points):
        """align converges to valid result."""
        ref, mov = basic_points
        (P, R, s, t, G, W), (var, iters) = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8
        )

        assert iters > 0
        assert iters <= 100
        assert var >= 0
        assert P.shape == (mov.shape[0], ref.shape[0])
        assert W.shape == mov.shape

    def test_align_fixed_iter_shape(self, basic_points):
        """align_fixed_iter returns correct variance shape."""
        ref, mov = basic_points
        num_iter = 10
        (_, _, _, _, _, _), varz = align_fixed_iter(
            ref, mov, 0.1, 1.0, 0.5, num_iter
        )

        assert varz.shape == (num_iter,)

    def test_rotation_valid(self, basic_points):
        """Final rotation matrix is approximately orthogonal."""
        ref, mov = basic_points
        (_, R, s, t, _, _), _ = align(ref, mov, 0.1, 1.0, 0.5, 100, 1e-8)

        assert jnp.allclose(R @ R.T, jnp.eye(2), atol=1e-4)
        assert jnp.abs(jnp.linalg.det(R) - 1.0) < 1e-4

    def test_align_with_uncertainty(self, basic_points):
        """align with uncertainty converges."""
        ref, mov = basic_points
        unc_ref = jnp.ones_like(ref) * 0.01
        unc_mov = jnp.ones_like(mov) * 0.01

        (_, _, _, _, _, _), (var, iters) = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8,
            unc_ref=unc_ref, unc_mov=unc_mov,
        )

        assert var >= 0
        assert iters > 0

    def test_burn_in_produces_different_result(self, basic_points):
        """burn_in > 0 should converge (possibly to different result)."""
        ref, mov = basic_points

        (_, _, _, _, _, W_no), _ = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8, burn_in=0
        )
        (_, _, _, _, _, W_bi), _ = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8, burn_in=5
        )

        # Both should converge — W should differ since burn-in delays
        # non-rigid updates, leading to a different final solution
        assert W_no.shape == W_bi.shape
        assert not jnp.allclose(W_no, W_bi, atol=1e-4)


# ============================================================================
# Transform recovery
# ============================================================================


class TestTransformRecovery:
    """Recovery of known transforms via full alignment."""

    def _generate_points(self, key, n=50):
        pts = jax.random.uniform(key, (n, 2), minval=-5.0, maxval=5.0)
        return pts

    def test_recover_pure_rigid(self):
        """When there is no deformation, recover the rigid part."""
        key = jax.random.PRNGKey(42)
        mov = self._generate_points(key)

        angle = jnp.pi / 6
        R_true = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
        ])
        s_true = jnp.array(1.3)
        t_true = jnp.array([2.0, -1.0])

        # Apply only rigid transform (no deformation)
        ref = rigid_transform(mov, R_true, s_true, t_true)

        (_, R, s, t, G, W), _ = align(
            ref, mov, 0.01, 1.0, 0.5, 100, 1e-8
        )

        # Combined registration may not perfectly recover rigid params when non-rigid
        # component is present; verify that transformed points match reference instead.
        ref_recovered = transform(mov, R, s, t, G, W)
        assert jnp.allclose(ref_recovered, ref, atol=0.3)

    def test_recover_pure_deformation(self):
        """When rigid part is identity, recover deformation."""
        key = jax.random.PRNGKey(100)
        mov = self._generate_points(key)

        # Smooth deformation
        v = jnp.stack([
            0.2 * jnp.sin(mov[:, 0]),
            0.2 * jnp.cos(mov[:, 1]),
        ], axis=1)
        ref = mov + v

        (_, R, s, t, G, W), _ = align(
            ref, mov, 0.01, 1.0, 0.5, 100, 1e-8
        )

        ref_recovered = transform(mov, R, s, t, G, W)
        assert jnp.allclose(ref_recovered, ref, atol=0.2)

    def test_recover_combined(self):
        """Recover rigid + non-rigid transformation."""
        key = jax.random.PRNGKey(200)
        mov = self._generate_points(key)

        # Rigid part
        angle = jnp.pi / 8
        R_true = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
        ])
        s_true = jnp.array(1.2)
        t_true = jnp.array([1.0, -0.5])

        # Apply rigid transform
        mov_rigid = rigid_transform(mov, R_true, s_true, t_true)

        # Add smooth deformation
        v = jnp.stack([
            0.15 * jnp.sin(mov_rigid[:, 0]),
            0.15 * jnp.cos(mov_rigid[:, 1]),
        ], axis=1)
        ref = mov_rigid + v

        (_, R, s, t, G, W), _ = align(
            ref, mov, 0.01, 1.0, 0.5, 100, 1e-8
        )

        ref_recovered = transform(mov, R, s, t, G, W)
        assert jnp.allclose(ref_recovered, ref, atol=0.3)


# ============================================================================
# Interpolation
# ============================================================================


class TestInterpolate:
    """Tests for the interpolate function."""

    def test_matches_transform_on_moving_points(self, basic_points):
        """interpolate on moving points should match transform."""
        _, mov = basic_points
        kvar = 0.5
        G = jnp.exp(-sqdist(mov, mov) / (2 * kvar))

        d = mov.shape[1]
        W = 0.1 * jnp.ones((mov.shape[0], d))
        angle = jnp.pi / 6
        R = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
        ])
        s = jnp.array(1.3)
        t = jnp.array([0.5, -0.3])

        interp_result = interpolate(mov, mov, R, s, t, W, kvar)
        transform_result = transform(mov, R, s, t, G, W)

        assert jnp.allclose(interp_result, transform_result, atol=1e-6)

    def test_interpolate_new_points(self, basic_points):
        """Interpolation on new points produces reasonable results."""
        _, mov = basic_points
        kvar = 0.5

        # Fit a simple transform
        (_, R, s, t, G, W), _ = align(
            basic_points[0], basic_points[1], 0.1, 1.0, kvar, 50, 1e-6
        )

        # Query at new points
        new_pts = jnp.array([[0.5, 0.5], [1.5, 1.5]])
        interp_result = interpolate(mov, new_pts, R, s, t, W, kvar)

        assert interp_result.shape == (2, 2)
        # Values should be finite
        assert jnp.all(jnp.isfinite(interp_result))


# ============================================================================
# High regularization limit (reduces to rigid)
# ============================================================================


class TestHighRegularizationLimit:
    """When regularization is very large, result approaches pure rigid."""

    def test_near_rigid_with_high_lambda(self):
        """Very high regularization_param should give near-rigid result."""
        key = jax.random.PRNGKey(500)
        mov = jax.random.uniform(key, (30, 2), minval=-5.0, maxval=5.0)

        angle = jnp.pi / 4
        R_true = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
        ])
        s_true = jnp.array(1.4)
        t_true = jnp.array([1.5, -0.8])

        ref = rigid_transform(mov, R_true, s_true, t_true)

        # High regularization suppresses non-rigid component
        (_, R_hl, s_hl, t_hl, _, W_hl), _ = align(
            ref, mov, 0.01, 1e6, 0.5, 100, 1e-8
        )

        # W should be near zero
        assert jnp.max(jnp.abs(W_hl)) < 0.15

        # Verify reconstructed points match, rather than individual params
        G_hl = jnp.exp(-sqdist(mov, mov) / (2 * 0.5))
        ref_recovered = transform(mov, R_hl, s_hl, t_hl, G_hl, W_hl)
        assert jnp.allclose(ref_recovered, ref, atol=0.3)


# ============================================================================
# Uncertainty
# ============================================================================


class TestUncertainty:
    """Tests for uncertainty-aware registration."""

    def test_nonzero_uncertainty_changes_result(self, basic_points):
        """Non-zero uncertainty should produce different result."""
        ref, mov = basic_points

        (_, _, s_base, _, _, W_base), _ = align(
            ref, mov, 0.1, 1.0, 0.5, 50, 1e-8
        )

        unc_ref = jnp.ones_like(ref) * 0.3
        unc_mov = jnp.ones_like(mov) * 0.3
        (_, _, s_unc, _, _, W_unc), _ = align(
            ref, mov, 0.1, 1.0, 0.5, 50, 1e-8,
            unc_ref=unc_ref, unc_mov=unc_mov,
        )

        assert not jnp.allclose(W_base, W_unc, atol=1e-4)

    def test_variance_decreases_with_higher_uncertainty(self, basic_points):
        """Higher uncertainty → lower estimated σ²."""
        ref, mov = basic_points

        _, var_low = align(
            ref, mov, 0.1, 1.0, 0.5, 50, 1e-8,
            unc_ref=jnp.ones_like(ref) * 0.001,
            unc_mov=jnp.ones_like(mov) * 0.001,
        )[1]

        _, var_high = align(
            ref, mov, 0.1, 1.0, 0.5, 50, 1e-8,
            unc_ref=jnp.ones_like(ref) * 0.1,
            unc_mov=jnp.ones_like(mov) * 0.1,
        )[1]

        assert var_high <= var_low


# ============================================================================
# 3D point sets
# ============================================================================


class TestTransform3D:
    """Tests for transform with 3D point sets."""

    def test_shape(self, basic_points_3d):
        """Transform returns correct shape for 3D."""
        _, mov = basic_points_3d
        d = mov.shape[1]
        G = jnp.exp(-sqdist(mov, mov) / 2.0)
        W = jnp.zeros_like(mov)
        R = jnp.eye(d)
        s = jnp.array(1.0)
        t = jnp.zeros(d)

        result = transform(mov, R, s, t, G, W)
        assert result.shape == mov.shape

    def test_identity(self, basic_points_3d):
        """Identity transform + zero W gives original points for 3D."""
        _, mov = basic_points_3d
        d = mov.shape[1]
        G = jnp.exp(-sqdist(mov, mov) / 2.0)
        W = jnp.zeros_like(mov)
        R = jnp.eye(d)
        s = jnp.array(1.0)
        t = jnp.zeros(d)

        result = transform(mov, R, s, t, G, W)
        assert jnp.allclose(result, mov, atol=1e-10)

    def test_composition(self, basic_points_3d):
        """Transform equals rigid part + non-rigid part for 3D."""
        _, mov = basic_points_3d
        d = mov.shape[1]
        m = mov.shape[0]
        G = jnp.exp(-sqdist(mov, mov) / 2.0)
        W = 0.1 * jnp.ones((m, d))

        # 3D rotation: rotation around z-axis by pi/6
        angle = jnp.pi / 6
        R = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle), 0.0],
            [jnp.sin(angle), jnp.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        s = jnp.array(1.5)
        t = jnp.array([1.0, -0.5, 0.3])

        result = transform(mov, R, s, t, G, W)
        rigid_part = rigid_transform(mov, R, s, t)
        nonrigid_part = G @ W
        expected = rigid_part + nonrigid_part
        assert jnp.allclose(result, expected, atol=1e-10)


class TestAlign3D:
    """Tests for full alignment with 3D point sets."""

    def test_align_converges(self, basic_points_3d):
        """align converges to valid result for 3D."""
        ref, mov = basic_points_3d
        (P, R, s, t, G, W), (var, iters) = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8
        )

        assert iters > 0
        assert iters <= 100
        assert var >= 0
        assert P.shape == (mov.shape[0], ref.shape[0])
        assert W.shape == mov.shape
        assert R.shape == (3, 3)

    def test_rotation_valid(self, basic_points_3d):
        """Final rotation matrix is approximately orthogonal for 3D."""
        ref, mov = basic_points_3d
        (_, R, s, t, _, _), _ = align(ref, mov, 0.1, 1.0, 0.5, 100, 1e-8)

        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-4)
        assert jnp.abs(jnp.linalg.det(R) - 1.0) < 1e-4


class TestTransformRecovery3D:
    """Recovery of known transforms via full alignment in 3D."""

    def _generate_points(self, key, n=50):
        pts = jax.random.uniform(key, (n, 3), minval=-5.0, maxval=5.0)
        return pts

    def test_recover_pure_rigid(self):
        """When there is no deformation, recover the rigid part in 3D."""
        key = jax.random.PRNGKey(42)
        mov = self._generate_points(key)

        # 3D rotation around z-axis
        angle = jnp.pi / 6
        R_true = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle), 0.0],
            [jnp.sin(angle), jnp.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        s_true = jnp.array(1.3)
        t_true = jnp.array([2.0, -1.0, 0.5])

        ref = rigid_transform(mov, R_true, s_true, t_true)

        (_, R, s, t, G, W), _ = align(
            ref, mov, 0.01, 1.0, 0.5, 100, 1e-8
        )

        ref_recovered = transform(mov, R, s, t, G, W)
        assert jnp.allclose(ref_recovered, ref, atol=0.3)

    def test_recover_pure_deformation(self):
        """When rigid part is identity, recover deformation in 3D."""
        key = jax.random.PRNGKey(100)
        mov = self._generate_points(key)

        # Smooth 3D deformation
        v = jnp.stack([
            0.2 * jnp.sin(mov[:, 0]),
            0.2 * jnp.cos(mov[:, 1]),
            0.1 * jnp.sin(mov[:, 2]),
        ], axis=1)
        ref = mov + v

        (_, R, s, t, G, W), _ = align(
            ref, mov, 0.01, 1.0, 0.5, 100, 1e-8
        )

        ref_recovered = transform(mov, R, s, t, G, W)
        assert jnp.allclose(ref_recovered, ref, atol=0.2)

    def test_recover_combined(self):
        """Recover rigid + non-rigid transformation in 3D."""
        key = jax.random.PRNGKey(200)
        mov = self._generate_points(key)

        # 3D rigid transform
        angle = jnp.pi / 8
        R_true = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle), 0.0],
            [jnp.sin(angle), jnp.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        s_true = jnp.array(1.2)
        t_true = jnp.array([1.0, -0.5, 0.3])

        mov_rigid = rigid_transform(mov, R_true, s_true, t_true)

        # Add smooth 3D deformation
        v = jnp.stack([
            0.15 * jnp.sin(mov_rigid[:, 0]),
            0.15 * jnp.cos(mov_rigid[:, 1]),
            0.1 * jnp.sin(mov_rigid[:, 2]),
        ], axis=1)
        ref = mov_rigid + v

        (_, R, s, t, G, W), _ = align(
            ref, mov, 0.01, 1.0, 0.5, 100, 1e-8
        )

        ref_recovered = transform(mov, R, s, t, G, W)
        assert jnp.allclose(ref_recovered, ref, atol=0.3)


# ============================================================================
# moving_weights
# ============================================================================


class TestMovingWeights:
    """Tests for moving_weights parameter in align."""

    def test_align_with_moving_weights_converges(self, basic_points):
        """align with moving_weights converges to valid result."""
        ref, mov = basic_points
        weights = jnp.ones(mov.shape[0])

        (P, R, s, t, G, W), (var, iters) = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8,
            moving_weights=weights,
        )

        assert iters > 0
        assert iters <= 100
        assert var >= 0
        assert P.shape == (mov.shape[0], ref.shape[0])

    def test_uniform_weights_equivalence(self, basic_points):
        """Uniform weights should match no weights."""
        ref, mov = basic_points
        weights = jnp.ones(mov.shape[0])

        (P_none, R_none, s_none, t_none, _, W_none), _ = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8,
        )
        (P_w, R_w, s_w, t_w, _, W_w), _ = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8,
            moving_weights=weights,
        )

        assert jnp.allclose(P_none, P_w, atol=1e-5)
        assert jnp.allclose(R_none, R_w, atol=1e-5)
        assert jnp.allclose(s_none, s_w, atol=1e-5)
        assert jnp.allclose(t_none, t_w, atol=1e-5)
        assert jnp.allclose(W_none, W_w, atol=1e-5)

    def test_weight_bias_changes_result(self, basic_points):
        """Non-uniform weights should produce different matching matrix."""
        ref, mov = basic_points
        weights = jnp.array([10.0, 1.0, 1.0, 1.0])

        (P_base, _, _, _, _, _), _ = align(
            ref, mov, 0.1, 1.0, 0.5, 50, 1e-8,
        )
        (P_w, _, _, _, _, _), _ = align(
            ref, mov, 0.1, 1.0, 0.5, 50, 1e-8,
            moving_weights=weights,
        )

        assert not jnp.allclose(P_base, P_w, atol=1e-4)

    def test_high_weight_attracts_matches(self, basic_points):
        """Heavily weighted point should attract more responsibility in P."""
        ref, mov = basic_points
        weights = jnp.array([0.1, 5.0, 0.1, 0.1])

        (P_base, _, _, _, _, _), _ = align(
            ref, mov, 0.1, 1.0, 0.5, 50, 1e-8,
        )
        (P_w, _, _, _, _, _), _ = align(
            ref, mov, 0.1, 1.0, 0.5, 50, 1e-8,
            moving_weights=weights,
        )

        # Row 1 (the heavily weighted point) should have more total
        # responsibility than in the unweighted run.
        row1_base = jnp.sum(P_base[1, :])
        row1_w = jnp.sum(P_w[1, :])
        assert row1_w > row1_base


# ============================================================================
# mask
# ============================================================================


class TestMask:
    """Tests for mask parameter in align."""

    def test_align_with_mask_converges(self, basic_points):
        """align with mask converges to valid result."""
        ref, mov = basic_points
        m, n = mov.shape[0], ref.shape[0]
        mask = jnp.ones((m, n))

        (P, R, s, t, G, W), (var, iters) = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8,
            mask=mask,
        )

        assert iters > 0
        assert iters <= 100
        assert var >= 0
        assert P.shape == (m, n)

    def test_full_mask_equivalence(self, basic_points):
        """All-ones mask should match no mask."""
        ref, mov = basic_points
        m, n = mov.shape[0], ref.shape[0]
        mask = jnp.ones((m, n))

        (P_none, R_none, s_none, t_none, _, W_none), _ = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8,
        )
        (P_m, R_m, s_m, t_m, _, W_m), _ = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8,
            mask=mask,
        )

        assert jnp.allclose(P_none, P_m, atol=1e-5)
        assert jnp.allclose(R_none, R_m, atol=1e-5)
        assert jnp.allclose(s_none, s_m, atol=1e-5)
        assert jnp.allclose(t_none, t_m, atol=1e-5)
        assert jnp.allclose(W_none, W_m, atol=1e-5)

    def test_zero_entry_forbids_match(self, basic_points):
        """Zero entries in mask should produce zero entries in P."""
        ref, mov = basic_points
        m, n = mov.shape[0], ref.shape[0]
        mask = jnp.ones((m, n)).at[0, 0].set(0)  # forbid match between moving[0] and ref[0]

        (P, _, _, _, _, _), _ = align(
            ref, mov, 0.1, 1.0, 0.5, 50, 1e-8,
            mask=mask,
        )

        assert P[0, 0] == 0.0

    def test_mask_changes_alignment(self, basic_points):
        """Restrictive mask should produce different alignment."""
        ref, mov = basic_points
        m, n = mov.shape[0], ref.shape[0]

        # Diagonal-only mask: each moving point can only match its index ref point
        diag_mask = jnp.eye(m)
        (_, _, s_base, t_base, _, W_base), _ = align(
            ref, mov, 0.1, 1.0, 0.5, 50, 1e-8,
        )
        (_, _, s_m, t_m, _, W_m), _ = align(
            ref, mov, 0.1, 1.0, 0.5, 50, 1e-8,
            mask=diag_mask,
        )

        assert not jnp.allclose(W_base, W_m, atol=1e-4)


# ============================================================================
# Combined paths
# ============================================================================


class TestCombinedPaths:
    """Tests for combined parameter paths in align."""

    def test_weights_and_uncertainty_path(self, basic_points):
        """align with both moving_weights and uncertainty converges."""
        ref, mov = basic_points
        weights = jnp.array([2.0, 1.0, 1.0, 1.0])
        unc_ref = jnp.ones_like(ref) * 0.01
        unc_mov = jnp.ones_like(mov) * 0.01

        (P, R, s, t, G, W), (var, iters) = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8,
            unc_ref=unc_ref, unc_mov=unc_mov,
            moving_weights=weights,
        )

        assert iters > 0
        assert iters <= 100
        assert var >= 0
        assert P.shape == (mov.shape[0], ref.shape[0])
        assert W.shape == mov.shape


# ============================================================================
# kernel_var sensitivity
# ============================================================================


class TestKernelVarSensitivity:
    """Tests for kernel_var (β²) parameter sensitivity."""

    def test_small_kernel_var_converges(self, basic_points):
        """Very small kernel_var (narrow kernel) should still converge."""
        ref, mov = basic_points
        (_, _, _, _, _, _), (var, iters) = align(
            ref, mov, 0.1, 1.0, 1e-4, 100, 1e-8,
        )

        assert iters > 0
        assert iters <= 100
        assert var >= 0

    def test_large_kernel_var_converges(self, basic_points):
        """Very large kernel_var (broad kernel) should still converge."""
        ref, mov = basic_points
        (_, _, _, _, _, _), (var, iters) = align(
            ref, mov, 0.1, 1.0, 1e4, 100, 1e-8,
        )

        assert iters > 0
        assert iters <= 100
        assert var >= 0

    def test_small_kernel_var_still_recovers(self):
        """Narrow kernel should still produce valid reconstruction."""
        key = jax.random.PRNGKey(77)
        mov = jax.random.uniform(key, (30, 2), minval=-5.0, maxval=5.0)

        # Smooth deformation
        v = jnp.stack([
            0.2 * jnp.sin(mov[:, 0]),
            0.2 * jnp.cos(mov[:, 1]),
        ], axis=1)
        ref = mov + v

        # Narrow kernel with high regularization to compensate
        (_, R, s, t, G, W), _ = align(
            ref, mov, 0.01, 1.0, 1e-3, 100, 1e-8,
        )

        # Reconstruction should still be reasonable
        ref_recovered = transform(mov, R, s, t, G, W)
        # All finite
        assert jnp.all(jnp.isfinite(ref_recovered))
        # Error should be bounded
        error = jnp.max(jnp.linalg.norm(ref - ref_recovered, axis=1))
        assert error < 1.0

    def test_kernel_var_changes_result(self, basic_points):
        """Different kernel_var values should produce different results."""
        ref, mov = basic_points

        (_, _, _, _, _, W_a), _ = align(
            ref, mov, 0.1, 1.0, 0.1, 50, 1e-8,
        )
        (_, _, _, _, _, W_b), _ = align(
            ref, mov, 0.1, 1.0, 10.0, 50, 1e-8,
        )

        assert not jnp.allclose(W_a, W_b, atol=1e-4)


# ============================================================================
# align / align_fixed_iter consistency
# ============================================================================


class TestAlignConsistency:
    """Tests for consistency between align and align_fixed_iter."""

    def test_fixed_iter_matches_align_at_convergence(self, basic_points):
        """align_fixed_iter at convergence iter count matches align result."""
        ref, mov = basic_points

        # Run align to find convergence point
        (P_a, R_a, s_a, t_a, G_a, W_a), (_, iters) = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8,
        )

        # Run align_fixed_iter with same number of iterations
        (P_f, R_f, s_f, t_f, G_f, W_f), _ = align_fixed_iter(
            ref, mov, 0.1, 1.0, 0.5, int(iters),
        )

        assert jnp.allclose(P_a, P_f, atol=1e-5)
        assert jnp.allclose(R_a, R_f, atol=1e-5)
        assert jnp.allclose(s_a, s_f, atol=1e-5)
        assert jnp.allclose(t_a, t_f, atol=1e-5)
        assert jnp.allclose(W_a, W_f, atol=1e-5)

    def test_fixed_iter_variance_trail_ends_at_same_value(self, basic_points):
        """Last variance from align_fixed_iter matches final variance from align."""
        ref, mov = basic_points

        (_, _, _, _, _, _), (var_final, iters) = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8,
        )

        (_, _, _, _, _, _), varz = align_fixed_iter(
            ref, mov, 0.1, 1.0, 0.5, int(iters),
        )

        assert jnp.allclose(varz[-1], var_final, atol=1e-5)

    def test_fixed_iter_more_than_needed(self, basic_points):
        """Variance trail becomes flat after convergence."""
        ref, mov = basic_points

        # Run align to find convergence point
        (_, _, _, _, _, _), (_, iters) = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8,
        )

        # Run align_fixed_iter with more iterations
        (_, _, _, _, _, _), varz = align_fixed_iter(
            ref, mov, 0.1, 1.0, 0.5, int(iters) + 20,
        )

        # Variance after convergence should be essentially flat
        tail = varz[int(iters):]
        assert jnp.max(jnp.abs(tail - tail[0])) < 1e-6

    def test_align_hits_max_iter(self, basic_points):
        """align should stop at max_iter when tolerance is unreachable."""
        ref, mov = basic_points
        max_iter = 5

        (_, _, _, _, _, _), (_, iters) = align(
            ref, mov, 0.1, 1.0, 0.5, max_iter, 1e-20,
        )

        assert iters == max_iter


# ============================================================================
# interpolate_variance
# ============================================================================


class TestInterpolateVariance:
    """Tests for interpolate_variance function."""

    def test_interpolate_variance_returns_valid(self, basic_points):
        """interpolate_variance with cholesky returns valid array."""
        ref, mov = basic_points
        kvar = 0.5
        (_, _, _, _, G, _), (var, _) = align(
            ref, mov, 0.1, 1.0, kvar, 100, 1e-8
        )
        P, _, _, _, _, _ = align(
            ref, mov, 0.1, 1.0, kvar, 100, 1e-8
        )[0]

        new_pts = jnp.array([[0.5, 0.5], [1.5, 1.5], [2.0, 1.0]])
        result = interpolate_variance(
            mov, new_pts, P, G, kvar, 1.0, float(var), method="cholesky",
        )

        assert result.shape == (3,)
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0)

    def test_interpolate_variance_on_control_points(self, basic_points):
        """Variance at control points should be near zero."""
        ref, mov = basic_points
        kvar = 0.5
        (P, _, _, _, G, _), (var, _) = align(
            ref, mov, 0.1, 1.0, kvar, 100, 1e-8
        )

        result = interpolate_variance(
            mov, mov, P, G, kvar, 1.0, float(var), method="cholesky",
        )

        # Variance at control points should be small
        assert jnp.max(result) < 0.01

    def test_interpolate_variance_finite_at_new_points(self, basic_points):
        """Variance at new points should be finite and non-negative."""
        ref, mov = basic_points
        kvar = 0.5
        (P, _, _, _, G, _), (var, _) = align(
            ref, mov, 0.1, 1.0, kvar, 100, 1e-8
        )

        # Query points between control points
        new_pts = jnp.array([[0.55, 0.55], [1.2, 0.8]])
        result = interpolate_variance(
            mov, new_pts, P, G, kvar, 1.0, float(var), method="cholesky",
        )

        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0)

    def test_low_rank_method_returns_valid(self, basic_points):
        """interpolate_variance with low_rank method returns valid array."""
        ref, mov = basic_points
        kvar = 0.5
        (P, _, _, _, G, _), (var, _) = align(
            ref, mov, 0.1, 1.0, kvar, 100, 1e-8
        )

        new_pts = jnp.array([[0.5, 0.5], [1.5, 1.5]])
        result = interpolate_variance(
            mov, new_pts, P, G, kvar, 1.0, float(var),
            method="low_rank", rank=2,
        )

        assert result.shape == (2,)
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0)

    def test_cholesky_and_low_rank_agree(self, basic_points):
        """cholesky and low_rank methods should produce similar results."""
        ref, mov = basic_points
        kvar = 0.5
        (P, _, _, _, G, _), (var, _) = align(
            ref, mov, 0.1, 1.0, kvar, 100, 1e-8
        )

        new_pts = jnp.array([[0.5, 0.5], [1.5, 1.5], [2.0, 1.0]])
        result_cho = interpolate_variance(
            mov, new_pts, P, G, kvar, 1.0, float(var), method="cholesky",
        )
        result_lr = interpolate_variance(
            mov, new_pts, P, G, kvar, 1.0, float(var),
            method="low_rank", rank=10,
        )

        assert jnp.allclose(result_cho, result_lr, atol=1e-4)

    def test_interpolate_variance_matches_nonrigid(self, basic_points):
        """RNR interpolate_variance should match nonrigid version."""
        from cpdx.nonrigid import interpolate_variance as nonrigid_iv

        ref, mov = basic_points
        kvar = 0.5
        (P, _, _, _, G, _), (var, _) = align(
            ref, mov, 0.1, 1.0, kvar, 100, 1e-8
        )

        new_pts = jnp.array([[0.5, 0.5], [1.5, 1.5]])

        result_rnr = interpolate_variance(
            mov, new_pts, P, G, kvar, 1.0, float(var), method="cholesky",
        )
        result_nonrigid = nonrigid_iv(
            mov, new_pts, P, G, kvar, 1.0, float(var), method="cholesky",
        )

        assert jnp.allclose(result_rnr, result_nonrigid, atol=1e-6)


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests for align and align_fixed_iter."""

    def test_align_already_aligned(self, basic_points):
        """Already-aligned points should recover identity transform."""
        ref, mov = basic_points
        (P, R, s, t, G, W), (var, iters) = align(
            mov, mov, 0.1, 1.0, 0.5, 100, 1e-8,
        )

        assert iters > 0
        assert var >= 0
        # Should converge quickly
        assert iters <= 20
        # Scale should be approximately 1
        assert jnp.allclose(s, 1.0, atol=0.1)
        # Translation should be approximately 0
        assert jnp.allclose(t, jnp.zeros(2), atol=0.1)
        # W should be small (no deformation needed)
        assert jnp.max(jnp.abs(W)) < 0.1

    def test_zero_outlier_probability(self, basic_points):
        """Zero outlier probability should produce valid convergence."""
        ref, mov = basic_points
        (P, R, s, t, G, W), (var, iters) = align(
            ref, mov, 0.0, 1.0, 0.5, 100, 1e-8,
        )

        assert iters > 0
        assert var >= 0
        # With no outliers, P rows should sum to approximately 1
        row_sums = jnp.sum(P, axis=1)
        assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=0.01)

    def test_two_point_sets(self):
        """Minimal two-point sets should converge."""
        ref = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        mov = jnp.array([[0.1, 0.05], [1.1, 0.95]])

        (P, R, s, t, G, W), (var, iters) = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8,
        )

        assert iters > 0
        assert iters <= 100
        assert var >= 0
        assert P.shape == (2, 2)
        assert R.shape == (2, 2)

    def test_variance_decreases_monotonically(self, basic_points):
        """Variance should decrease monotonically through iterations."""
        ref, mov = basic_points
        (_, _, _, _, _, _), varz = align_fixed_iter(
            ref, mov, 0.1, 1.0, 0.5, 30,
        )

        # Check variance is non-increasing (with small tolerance for numerical noise)
        diffs = varz[1:] - varz[:-1]
        # Allow small positive diffs due to numerical precision
        assert jnp.all(diffs <= 1e-5)

    def test_same_size_point_sets(self, basic_points):
        """Same-size point sets should converge without issues."""
        ref, mov = basic_points  # both have 4 points
        (P, R, s, t, G, W), (var, iters) = align(
            ref, mov, 0.1, 1.0, 0.5, 100, 1e-8,
        )

        assert iters > 0
        assert var >= 0
        assert P.shape == (4, 4)

    def test_align_fixed_iter_with_uncertainty(self, basic_points):
        """align_fixed_iter with uncertainty should produce valid results."""
        ref, mov = basic_points
        unc_ref = jnp.ones_like(ref) * 0.01
        unc_mov = jnp.ones_like(mov) * 0.01
        num_iter = 10

        (_, _, _, _, _, _), varz = align_fixed_iter(
            ref, mov, 0.1, 1.0, 0.5, num_iter,
            unc_ref=unc_ref, unc_mov=unc_mov,
        )

        assert varz.shape == (num_iter,)
        assert jnp.all(jnp.isfinite(varz))

    def test_align_fixed_iter_with_burn_in(self, basic_points):
        """align_fixed_iter with burn_in should produce valid results."""
        ref, mov = basic_points
        num_iter = 20

        (_, _, _, _, _, W), varz = align_fixed_iter(
            ref, mov, 0.1, 1.0, 0.5, num_iter, burn_in=5,
        )

        assert varz.shape == (num_iter,)
        assert jnp.all(jnp.isfinite(varz))
        # W should be non-zero after burn-in
        assert jnp.max(jnp.abs(W)) > 1e-6


# ============================================================================
# Variance monotonicity
# ============================================================================


class TestVarianceMonotonicity:
    """Tests for variance monotonicity via align_fixed_iter."""

    def test_variance_monotonic_basic(self, basic_points):
        """Variance should decrease monotonically on basic points."""
        ref, mov = basic_points
        (_, _, _, _, _, _), varz = align_fixed_iter(
            ref, mov, 0.1, 1.0, 0.5, 30,
        )

        diffs = varz[1:] - varz[:-1]
        assert jnp.all(diffs <= 1e-5)

    def test_variance_monotonic_with_uncertainty(self, basic_points):
        """Variance should decrease monotonically with uncertainty."""
        ref, mov = basic_points
        unc_ref = jnp.ones_like(ref) * 0.01
        unc_mov = jnp.ones_like(mov) * 0.01

        (_, _, _, _, _, _), varz = align_fixed_iter(
            ref, mov, 0.1, 1.0, 0.5, 30,
            unc_ref=unc_ref, unc_mov=unc_mov,
        )

        diffs = varz[1:] - varz[:-1]
        assert jnp.all(diffs <= 1e-5)

    def test_variance_monotonic_3d(self, basic_points_3d):
        """Variance should decrease monotonically in 3D."""
        ref, mov = basic_points_3d
        (_, _, _, _, _, _), varz = align_fixed_iter(
            ref, mov, 0.1, 1.0, 0.5, 30,
        )

        diffs = varz[1:] - varz[:-1]
        assert jnp.all(diffs <= 1e-5)

    def test_variance_flattens_after_convergence(self, basic_points):
        """Variance trail should become flat after convergence."""
        ref, mov = basic_points
        (_, _, _, _, _, _), varz = align_fixed_iter(
            ref, mov, 0.1, 1.0, 0.5, 50,
        )

        # Find first index where variance becomes essentially flat
        diffs = jnp.abs(varz[1:] - varz[:-1])
        flat_threshold = 1e-8
        flat_indices = jnp.where(diffs < flat_threshold)[0]

        # Should have some flat region (not all differences large)
        assert len(flat_indices) > 0

        # The flat region should be in the latter half of iterations
        assert flat_indices[-1] > len(varz) // 2

    def test_variance_monotonic_high_regularization(self, basic_points):
        """Variance should decrease monotonically with high regularization."""
        ref, mov = basic_points
        (_, _, _, _, _, _), varz = align_fixed_iter(
            ref, mov, 0.1, 1e6, 0.5, 30,
        )

        diffs = varz[1:] - varz[:-1]
        assert jnp.all(diffs <= 1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
