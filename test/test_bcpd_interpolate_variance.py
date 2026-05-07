"""Tests for BCPD interpolation variance and inverse transformation uncertainty."""

import jax.numpy as jnp
import numpy as np
import pytest

from cpdx.bayes import align
from cpdx.bayes.kernel import gaussian_kernel
from cpdx.bayes.util import _jacobian_forward
from cpdx.bayes.util import interpolate
from cpdx.bayes.util import interpolate_variance
from cpdx.bayes.util import interpolate_variance_inverse
from cpdx.bayes.util import invert_gp_bcpd_mapping
from cpdx.bayes.util import residual


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
    """Run BCPD alignment and return parameters."""
    mov, ref = simple_point_sets
    kernel = gaussian_kernel
    beta = 0.5
    lambda_ = 1.0
    gamma = 1.0
    kappa = np.inf

    (P, R, s, t, v), (var, _) = align(
        ref,
        mov,
        outlier_prob=0.01,
        num_iter=50,
        tolerance=1e-6,
        kernel=kernel,
        lambda_param=lambda_,
        kernel_beta=beta,
        gamma=gamma,
        kappa=kappa,
        transform_mode="nonrigid",
    )

    # Compute residual for interpolation functions
    resid = residual(ref, mov, P, R, s, t)
    G = jnp.exp(
        -jnp.sum((mov[:, None, :] - mov[None, :, :]) ** 2, axis=2) / (2 * beta)
    )

    return mov, ref, P, R, s, t, v, G, resid, var, lambda_, beta, kernel


class TestInterpolateVariance:
    """Tests for interpolate_variance function."""

    def test_variance_positive(self, aligned_params):
        """Variance values should be non-negative."""
        mov, ref, P, R, s, t, v, G, resid, var, lambda_, beta, kernel = (
            aligned_params
        )
        key = np.random.RandomState(123)
        interp = key.randn(10, 2)

        variance = interpolate_variance(
            mov, interp, P, G, kernel, beta, s, lambda_, var
        )
        assert variance.shape == (10,)
        assert jnp.all(variance >= 0)

    def test_variance_at_control_points(self, aligned_params):
        """Variance should be small at the original moving points."""
        mov, ref, P, R, s, t, v, G, resid, var, lambda_, beta, kernel = (
            aligned_params
        )

        variance = interpolate_variance(
            mov, mov, P, G, kernel, beta, s, lambda_, var
        )
        assert variance.shape == (mov.shape[0],)
        assert jnp.all(variance >= 0)
        # Variance at control points should be relatively small
        assert jnp.max(variance) < 1.0

    def test_variance_increases_with_distance(self, aligned_params):
        """BCPD variance approaches K(z,z)/lambda as query points move far from control points,
        because the kernel vector k(z) becomes small and the information term vanishes.
        """
        mov, ref, P, R, s, t, v, G, resid, var, lambda_, beta, kernel = (
            aligned_params
        )

        # Points near control points
        rng = np.random.RandomState(456)
        near = mov[:5] + rng.randn(5, 2) * 0.01
        # Points far from control points
        rng2 = np.random.RandomState(789)
        far = mov[:5] + rng2.randn(5, 2) * 5.0

        var_near = interpolate_variance(
            mov, near, P, G, kernel, beta, s, lambda_, var
        )
        var_far = interpolate_variance(
            mov, far, P, G, kernel, beta, s, lambda_, var
        )
        # Far points have small kernel values, so the information term vanishes
        # and variance approaches K(z,z)/lambda (the prior variance)
        assert jnp.mean(var_far) > jnp.mean(var_near)
        # Far variance should be close to 1/lambda
        assert jnp.allclose(jnp.mean(var_far), 1.0 / lambda_, atol=0.1)

    def test_variance_scales_with_lambda(self, aligned_params):
        """Higher lambda reduces psi, but the 1/lambda scaling dominates."""
        mov, ref, P, R, s, t, v, G, resid, var, lambda_, beta, kernel = (
            aligned_params
        )
        key = np.random.RandomState(42)
        interp = key.randn(5, 2)

        var_low = interpolate_variance(
            mov, interp, P, G, kernel, beta, s, 0.1, var
        )
        var_high = interpolate_variance(
            mov, interp, P, G, kernel, beta, s, 10.0, var
        )
        # Higher lambda = smaller 1/lambda factor = lower variance
        assert jnp.mean(var_high) < jnp.mean(var_low)


class TestJacobianForward:
    """Tests for the forward Jacobian computation."""

    def test_jacobian_shape(self, aligned_params):
        """Jacobian should have correct shape."""
        mov, ref, P, R, s, t, v, G, resid, var, lambda_, beta, kernel = (
            aligned_params
        )
        key = np.random.RandomState(42)
        z = key.randn(5, 2)

        J = _jacobian_forward(
            z, mov, resid, G, P, kernel, beta, s, lambda_, var
        )
        assert J.shape == (5, 2, 2)

    def test_jacobian_near_identity_for_small_residual(self, aligned_params):
        """When residual is small, Jacobian should be near identity."""
        mov, ref, P, R, s, t, v, G, resid, var, lambda_, beta, kernel = (
            aligned_params
        )
        key = np.random.RandomState(42)
        z = key.randn(3, 2)

        # Use very small residual
        resid_small = resid * 1e-6
        J = _jacobian_forward(
            z, mov, resid_small, G, P, kernel, beta, s, lambda_, var
        )
        for i in range(3):
            assert jnp.allclose(J[i], jnp.eye(2), atol=1e-5)

    def test_jacobian_matches_autodiff(self, aligned_params):
        """Jacobian should match JAX autodiff."""
        import jax

        mov, ref, P, R, s, t, v, G, resid, var, lambda_, beta, kernel = (
            aligned_params
        )
        z0 = jnp.array([0.5, 0.3])

        # Analytic Jacobian
        J_analytic = _jacobian_forward(
            z0[None, :], mov, resid, G, P, kernel, beta, s, lambda_, var
        )[0]

        # Autodiff Jacobian
        def T(z_pt):
            return (
                z_pt
                + interpolate(
                    mov,
                    z_pt[None, :],
                    resid,
                    P,
                    G,
                    kernel,
                    beta,
                    s,
                    lambda_,
                    var,
                )[0]
            )

        J_autodiff = jax.jacfwd(T)(z0)

        assert jnp.allclose(J_analytic, J_autodiff, atol=1e-4)


class TestInterpolateVarianceInverse:
    """Tests for inverse transformation uncertainty."""

    def test_inverse_covariance_shape(self, aligned_params):
        """Inverse covariance should have correct shape."""
        mov, ref, P, R, s, t, v, G, resid, var, lambda_, beta, kernel = (
            aligned_params
        )
        key = np.random.RandomState(42)
        target = key.randn(5, 2)

        z_hat, cov = interpolate_variance_inverse(
            mov, target, P, G, resid, kernel, beta, s, lambda_, var
        )
        assert z_hat.shape == (5, 2)
        assert cov.shape == (5, 2, 2)

    def test_inverse_covariance_positive_definite(self, aligned_params):
        """Inverse covariance matrices should be positive semi-definite."""
        mov, ref, P, R, s, t, v, G, resid, var, lambda_, beta, kernel = (
            aligned_params
        )
        key = np.random.RandomState(42)
        target = key.randn(5, 2)

        z_hat, cov = interpolate_variance_inverse(
            mov, target, P, G, resid, kernel, beta, s, lambda_, var
        )
        # Check eigenvalues are non-negative
        for i in range(5):
            eigvals = jnp.linalg.eigvalsh(cov[i])
            assert jnp.all(eigvals >= -1e-6)

    def test_inverse_roundtrip(self, aligned_params):
        """Inverting T(y_m) should return points near y_m."""
        mov, ref, P, R, s, t, v, G, resid, var, lambda_, beta, kernel = (
            aligned_params
        )

        # Transform the moving points: T(y) = y + v(y)
        target = mov + interpolate(
            mov, mov, resid, P, G, kernel, beta, s, lambda_, var
        )

        z_hat, cov = interpolate_variance_inverse(
            mov, target, P, G, resid, kernel, beta, s, lambda_, var
        )
        # Should be close to original moving points
        assert jnp.allclose(z_hat, mov, atol=1e-3)

    def test_inverse_covariance_trace(self, aligned_params):
        """For near-identity transforms, trace of inverse cov should be ~d * sigma_v^2."""
        mov, ref, P, R, s, t, v, G, resid, var, lambda_, beta, kernel = (
            aligned_params
        )
        key = np.random.RandomState(42)
        target = key.randn(3, 2)

        z_hat, cov = interpolate_variance_inverse(
            mov, target, P, G, resid, kernel, beta, s, lambda_, var
        )

        # Get forward variance at inverted points
        sigma_v_sq = interpolate_variance(
            mov, z_hat, P, G, kernel, beta, s, lambda_, var
        )

        # The trace should be approximately d * sigma_v^2
        trace_ratio = jnp.trace(cov[0]) / (2.0 * sigma_v_sq[0])
        assert (
            0.3 < trace_ratio < 3.0
        )  # Allow tolerance for non-identity transforms with compression/shear


class TestInvertGpBcpdMapping:
    """Verify that invert_gp_bcpd_mapping works correctly."""

    def test_invert_consistency(self, aligned_params):
        """invert_gp_bcpd_mapping should return points close to original."""
        mov, ref, P, R, s, t, v, G, resid, var, lambda_, beta, kernel = (
            aligned_params
        )

        # Transform moving points: T(y) = y + v(y)
        target = mov + interpolate(
            mov, mov, resid, P, G, kernel, beta, s, lambda_, var
        )

        # Invert should return points close to original
        z_hat = invert_gp_bcpd_mapping(
            target, mov, resid, P, G, kernel, beta, s, lambda_, var
        )
        assert jnp.allclose(z_hat, mov, atol=1e-3)
