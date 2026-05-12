"""Comprehensive roundtrip tests for invert_gp_mapping.

Verifies that for a forward GP mapping T with inverse T^{-1},
composing them returns the original points: T^{-1}(T(x)) = x.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from cpdx.nonrigid import align
from cpdx.nonrigid import interpolate
from cpdx.nonrigid import invert_gp_mapping
from cpdx.nonrigid import transform


# -- Fixtures --


@pytest.fixture
def aligned_params():
    """Run alignment on a moderate deformation and return parameters."""
    rng = np.random.RandomState(42)
    # Moving points on a circle
    angles = jnp.linspace(0, 2 * jnp.pi, 30, endpoint=False)
    mov = jnp.column_stack([jnp.cos(angles), jnp.sin(angles)])
    # Reference points: slightly deformed circle
    ref = mov * 1.2 + rng.randn(30, 2) * 0.15

    (P, G, W), (var, _) = align(
        ref,
        mov,
        outlier_prob=0.01,
        regularization_param=1.0,
        kernel_var=0.5,
        max_iter=100,
        tolerance=1e-8,
    )
    return mov, ref, P, G, W, var


@pytest.fixture
def near_identity_params():
    """Run alignment on nearly identical point sets (near-identity transform)."""
    rng = np.random.RandomState(99)
    angles = jnp.linspace(0, 2 * jnp.pi, 20, endpoint=False)
    mov = jnp.column_stack([jnp.cos(angles), jnp.sin(angles)])
    # Tiny perturbation
    ref = mov + rng.randn(20, 2) * 1e-4

    (P, G, W), (var, _) = align(
        ref,
        mov,
        outlier_prob=0.01,
        regularization_param=1.0,
        kernel_var=0.5,
        max_iter=100,
        tolerance=1e-8,
    )
    return mov, ref, P, G, W, var


@pytest.fixture
def large_deformation_params():
    """Run alignment with significant but invertible nonrigid deformation."""
    rng = np.random.RandomState(77)
    # Grid of control points
    xs, ys = jnp.meshgrid(jnp.linspace(-1, 1, 6), jnp.linspace(-1, 1, 6))
    mov = jnp.column_stack([xs.ravel(), ys.ravel()])

    # Apply a moderate bending deformation (smaller coefficients)
    ref = mov + jnp.column_stack(
        [
            0.2 * mov[:, 1] ** 2,  # bend in x depending on y
            0.15 * mov[:, 0] ** 2,  # bend in y depending on x
        ]
    )
    ref = ref + rng.randn(*ref.shape) * 0.02

    (P, G, W), (var, _) = align(
        ref,
        mov,
        outlier_prob=0.01,
        regularization_param=0.5,
        kernel_var=0.5,
        max_iter=100,
        tolerance=1e-8,
    )
    return mov, ref, P, G, W, var


# -- Helper --


def forward_map(x, mov, W, kernel_var):
    """Compute T(x) = x + interpolate(mov, x, W, kernel_var)."""
    return x + interpolate(mov, x, W, kernel_var)


# -- Tests --


class TestRoundtrip:
    """Tests for forward-then-inverse roundtrip correctness."""

    def test_roundtrip_control_points(self, aligned_params):
        """Inverting T(mov) should return mov with high accuracy.

        Control points are the strongest anchors of the GP field, so
        inversion tolerance should be as tight as (or tighter than)
        for arbitrary interpolated points.
        """
        mov, ref, P, G, W, var = aligned_params

        # Forward transform of control points
        y = transform(mov, G, W)

        # Invert
        x_recovered = invert_gp_mapping(
            y, mov, W, kernel_var=0.5, max_iter=20, tol=1e-9
        )

        # Control points should be recovered very accurately
        assert x_recovered.shape == mov.shape
        assert jnp.allclose(x_recovered, mov, atol=1e-5)

    def test_roundtrip_arbitrary_points(self, aligned_params):
        """Forward-map arbitrary source points, then invert.

        For x -> T(x) -> T^{-1}(T(x)), the result should equal x.
        """
        mov, ref, P, G, W, var = aligned_params

        # Generate random source points in the bounding box of mov
        rng = np.random.RandomState(123)
        lo = mov.min(axis=0) - 0.2
        hi = mov.max(axis=0) + 0.2
        x = lo + rng.rand(20, 2) * (hi - lo)
        x = jnp.array(x)

        # Forward map then invert
        y = forward_map(x, mov, W, kernel_var=0.5)
        x_recovered = invert_gp_mapping(
            y, mov, W, kernel_var=0.5, max_iter=20, tol=1e-9
        )

        mse = jnp.mean(jnp.square(x - x_recovered))
        assert mse < 1e-6

    def test_roundtrip_reference_points(self, aligned_params):
        """Inverting the reference points should return points near mov.

        After alignment, ref is approximately T(mov) under the learned
        transform. Since alignment uses soft matching with outliers,
        the correspondence is approximate and the inverse will not
        recover mov exactly. We check that the maximum displacement
        is bounded.
        """
        mov, ref, P, G, W, var = aligned_params

        x_recovered = invert_gp_mapping(
            ref, mov, W, kernel_var=0.5, max_iter=20, tol=1e-9
        )

        assert x_recovered.shape == mov.shape
        # Soft matching means ref is not exactly T(mov); tolerance is loose.
        # The ref fixture uses 20% scale + noise, so errors ~0.2-0.3 are expected.
        max_err = jnp.max(jnp.abs(x_recovered - mov))
        assert max_err < 0.3

    def test_roundtrip_near_identity(self, near_identity_params):
        """Near-identity transforms should invert to machine precision.

        When displacements are tiny, Newton-Raphson converges
        essentially in one step.
        """
        mov, ref, P, G, W, var = near_identity_params

        rng = np.random.RandomState(42)
        lo = mov.min(axis=0) - 0.1
        hi = mov.max(axis=0) + 0.1
        x = lo + rng.rand(10, 2) * (hi - lo)
        x = jnp.array(x)

        y = forward_map(x, mov, W, kernel_var=0.5)
        x_recovered = invert_gp_mapping(
            y, mov, W, kernel_var=0.5, max_iter=20, tol=1e-12
        )

        mse = jnp.mean(jnp.square(x - x_recovered))
        assert mse < 1e-8

    def test_roundtrip_large_deformation(self, large_deformation_params):
        """Inversion should work under strong nonrigid deformation.

        Uses low regularization to allow large displacements,
        stress-testing the Newton-Raphson solver.
        """
        mov, ref, P, G, W, var = large_deformation_params

        rng = np.random.RandomState(55)
        lo = mov.min(axis=0) - 0.1
        hi = mov.max(axis=0) + 0.1
        x = lo + rng.rand(15, 2) * (hi - lo)
        x = jnp.array(x)

        y = forward_map(x, mov, W, kernel_var=0.5)
        x_recovered = invert_gp_mapping(
            y, mov, W, kernel_var=0.5, max_iter=30, tol=1e-9
        )

        mse = jnp.mean(jnp.square(x - x_recovered))
        assert mse < 1e-3
