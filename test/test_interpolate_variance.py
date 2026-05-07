"""Tests for interpolation variance and inverse transformation uncertainty."""

import jax.numpy as jnp
import numpy as np
import pytest

from cpdx.nonrigid import _jacobian_forward
from cpdx.nonrigid import align
from cpdx.nonrigid import interpolate
from cpdx.nonrigid import interpolate_variance
from cpdx.nonrigid import interpolate_variance_inverse
from cpdx.nonrigid import invert_gp_mapping


@pytest.fixture
def simple_point_sets():
    """Create simple 2D point sets for testing."""
    key = np.random.RandomState(42)
    # Moving points on a circle
    angles = jnp.linspace(0, 2 * jnp.pi, 20, endpoint=False)
    mov = jnp.column_stack([jnp.cos(angles), jnp.sin(angles)])
    # Reference points: slightly deformed circle
    ref = mov * 1.1 + key.randn(20, 2) * 0.1
    return mov, ref


@pytest.fixture
def aligned_params(simple_point_sets):
    """Run alignment and return parameters."""
    mov, ref = simple_point_sets
    (P, G, W), (var, _) = align(
        ref,
        mov,
        outlier_prob=0.01,
        regularization_param=1.0,
        kernel_var=0.5,
        max_iter=50,
        tolerance=1e-6,
    )
    return mov, ref, P, G, W, var


class TestInterpolateVariance:
    """Tests for interpolate_variance function."""

    def test_variance_positive(self, aligned_params):
        """Variance values should be non-negative."""
        mov, ref, P, G, W, var = aligned_params
        key = np.random.RandomState(123)
        interp = key.randn(10, 2)

        variance = interpolate_variance(
            mov,
            interp,
            P,
            G,
            kernel_var=0.5,
            regularization_param=1.0,
            var=var,
            method="cholesky",
        )
        assert variance.shape == (10,)
        assert jnp.all(variance >= 0)

    def test_variance_at_control_points(self, aligned_params):
        """Variance should be small at the original moving points."""
        mov, ref, P, G, W, var = aligned_params

        variance = interpolate_variance(
            mov,
            mov,
            P,
            G,
            kernel_var=0.5,
            regularization_param=1.0,
            var=var,
            method="cholesky",
        )
        assert variance.shape == (mov.shape[0],)
        assert jnp.all(variance >= 0)
        # Variance at control points should be relatively small
        assert jnp.max(variance) < 1.0

    def test_variance_decreases_with_distance(self, aligned_params):
        """Variance k^T H^{-1} k decreases as query points move far from control points,
        because the kernel vector k(z) becomes small."""
        mov, ref, P, G, W, var = aligned_params

        # Points near control points
        rng = np.random.RandomState(456)
        near = mov[:5] + rng.randn(5, 2) * 0.01
        # Points far from control points
        rng2 = np.random.RandomState(789)
        far = mov[:5] + rng2.randn(5, 2) * 5.0

        var_near = interpolate_variance(
            mov,
            near,
            P,
            G,
            kernel_var=0.5,
            regularization_param=1.0,
            var=var,
            method="cholesky",
        )
        var_far = interpolate_variance(
            mov,
            far,
            P,
            G,
            kernel_var=0.5,
            regularization_param=1.0,
            var=var,
            method="cholesky",
        )
        # Far points have small kernel values, so k^T H^{-1} k is small
        assert jnp.mean(var_far) < jnp.mean(var_near)

    def test_variance_scales_with_lambda(self, aligned_params):
        """Higher lambda makes H larger, reducing H^{-1}, so variance decreases."""
        mov, ref, P, G, W, var = aligned_params
        key = np.random.RandomState(42)
        interp = key.randn(5, 2)

        var_low = interpolate_variance(
            mov,
            interp,
            P,
            G,
            kernel_var=0.5,
            regularization_param=0.1,
            var=var,
            method="cholesky",
        )
        var_high = interpolate_variance(
            mov,
            interp,
            P,
            G,
            kernel_var=0.5,
            regularization_param=10.0,
            var=var,
            method="cholesky",
        )
        # Higher lambda = larger H = smaller H^{-1} = lower variance
        assert jnp.mean(var_high) < jnp.mean(var_low)

    def test_cholesky_vs_low_rank(self, aligned_params):
        """Cholesky and low-rank methods should produce similar results."""
        mov, ref, P, G, W, var = aligned_params
        key = np.random.RandomState(42)
        interp = key.randn(10, 2)

        var_cholesky = interpolate_variance(
            mov,
            interp,
            P,
            G,
            kernel_var=0.5,
            regularization_param=1.0,
            var=var,
            method="cholesky",
        )
        var_lowrank = interpolate_variance(
            mov,
            interp,
            P,
            G,
            kernel_var=0.5,
            regularization_param=1.0,
            var=var,
            method="low_rank",
            rank=20,  # full rank for 20 control points
        )
        # Should be reasonably close; differences expected due to different
        # regularization (eps * I) applied in different bases
        assert jnp.allclose(var_cholesky, var_lowrank, rtol=0.05, atol=1e-4)


class TestJacobianForward:
    """Tests for the forward Jacobian computation."""

    def test_jacobian_shape(self, aligned_params):
        """Jacobian should have correct shape."""
        mov, ref, P, G, W, var = aligned_params
        key = np.random.RandomState(42)
        z = key.randn(5, 2)

        J = _jacobian_forward(z, mov, W, kernel_var=0.5)
        assert J.shape == (5, 2, 2)

    def test_jacobian_near_identity_for_small_weights(self, aligned_params):
        """When weights are small, Jacobian should be near identity."""
        mov, ref, P, G, W, var = aligned_params
        key = np.random.RandomState(42)
        z = key.randn(3, 2)

        # Use very small weights
        W_small = W * 1e-6
        J = _jacobian_forward(z, mov, W_small, kernel_var=0.5)  # type: ignore
        for i in range(3):
            assert jnp.allclose(J[i], jnp.eye(2), atol=1e-5)

    def test_jacobian_matches_autodiff(self, aligned_params):
        """Jacobian should match JAX autodiff."""
        import jax

        mov, ref, P, G, W, var = aligned_params
        z0 = jnp.array([0.5, 0.3])

        # Analytic Jacobian
        J_analytic = _jacobian_forward(z0[None, :], mov, W, kernel_var=0.5)[0]

        # Autodiff Jacobian
        def T(z_pt):
            return z_pt + interpolate(mov, z_pt[None, :], W, 0.5)[0]

        J_autodiff = jax.jacfwd(T)(z0)

        assert jnp.allclose(J_analytic, J_autodiff, atol=1e-5)


class TestInterpolateVarianceInverse:
    """Tests for inverse transformation uncertainty."""

    def test_inverse_covariance_shape(self, aligned_params):
        """Inverse covariance should have correct shape."""
        mov, ref, P, G, W, var = aligned_params
        key = np.random.RandomState(42)
        target = key.randn(5, 2)

        z_hat, cov = interpolate_variance_inverse(
            mov,
            target,
            P,
            G,
            W,
            kernel_var=0.5,
            regularization_param=1.0,
            var=var,
            method="cholesky",
        )
        assert z_hat.shape == (5, 2)
        assert cov.shape == (5, 2, 2)

    def test_inverse_covariance_positive_definite(self, aligned_params):
        """Inverse covariance matrices should be positive semi-definite."""
        mov, ref, P, G, W, var = aligned_params
        key = np.random.RandomState(42)
        target = key.randn(5, 2)

        z_hat, cov = interpolate_variance_inverse(
            mov,
            target,
            P,
            G,
            W,
            kernel_var=0.5,
            regularization_param=1.0,
            var=var,
            method="cholesky",
        )
        # Check eigenvalues are non-negative
        for i in range(5):
            eigvals = jnp.linalg.eigvalsh(cov[i])
            assert jnp.all(eigvals >= -1e-6)

    def test_inverse_roundtrip(self, aligned_params):
        """Inverting T(y_m) should return points near y_m."""
        mov, ref, P, G, W, var = aligned_params

        # Transform the moving points
        from cpdx.nonrigid import transform

        target = transform(mov, G, W)

        z_hat, cov = interpolate_variance_inverse(
            mov,
            target,
            P,
            G,
            W,
            kernel_var=0.5,
            regularization_param=1.0,
            var=var,
            method="cholesky",
        )
        # Should be close to original moving points
        assert jnp.allclose(z_hat, mov, atol=1e-3)

    def test_inverse_covariance_near_identity_transform(self, aligned_params):
        """For near-identity transforms, inverse cov should be ~sigma_v^2 * I."""
        mov, ref, P, G, W, var = aligned_params
        key = np.random.RandomState(42)
        target = key.randn(3, 2)

        z_hat, cov = interpolate_variance_inverse(
            mov,
            target,
            P,
            G,
            W,
            kernel_var=0.5,
            regularization_param=1.0,
            var=var,
            method="cholesky",
        )

        # Get forward variance at inverted points
        sigma_v_sq = interpolate_variance(
            mov,
            z_hat,
            P,
            G,
            kernel_var=0.5,
            regularization_param=1.0,
            var=var,
            method="cholesky",
        )

        # The trace should be approximately d * sigma_v^2
        trace_ratio = jnp.trace(cov[0]) / (2.0 * sigma_v_sq[0])
        assert (
            0.5 < trace_ratio < 2.0
        )  # Allow some tolerance for non-identity transforms


class TestInvertGpMappingRefactored:
    """Verify that refactored invert_gp_mapping still works correctly."""

    def test_invert_consistency(self, aligned_params):
        """Refactored invert_gp_mapping should match original behavior."""
        mov, ref, P, G, W, var = aligned_params

        # Transform moving points
        from cpdx.nonrigid import transform

        target = transform(mov, G, W)

        # Invert should return points close to original
        z_hat = invert_gp_mapping(target, mov, W, kernel_var=0.5)
        assert jnp.allclose(z_hat, mov, atol=1e-3)
