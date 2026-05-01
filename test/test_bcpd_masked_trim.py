"""Integration test: verify that trim_with_mask prevents NaN in BCPD alignment.

When a mask has all-zero rows or columns, the BCPD algorithm would produce NaN
values due to division by zero in the E-step. Using trim_with_mask to remove
unmatchable points before alignment should prevent this.
"""

import math

import jax.numpy as jnp

from cpdx.bayes import align as bcpd_align
from cpdx.bayes.kernel import gaussian_kernel
from cpdx.util import trim_with_mask


def test_bcpd_no_nan_with_trimmed_mask():
    """BCPD should not produce NaN when unmatchable points are trimmed.

    Create a mask with all-zero rows and columns, trim using trim_with_mask,
    then run BCPD alignment on the trimmed data.
    """
    # Reference points: 4 corners of a square
    ref = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],  # center point (will be unmatchable)
        ]
    )
    # Moving points: shifted square
    mov = jnp.array(
        [
            [0.1, 0.1],
            [1.1, 0.1],
            [1.1, 1.1],
            [0.1, 1.1],
            [2.0, 2.0],  # outlier (will be unmatchable)
        ]
    )

    # Mask: allow matching for first 4 points, but make row 4 and column 4 all zero
    # This means mov[4] cannot match any ref, and ref[4] cannot match any mov
    mask = jnp.array(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    # Trim unmatchable points
    trimmed_ref, trimmed_mov, trimmed_mask, mov_idx, ref_idx = trim_with_mask(
        ref, mov, mask
    )

    # Run BCPD on trimmed data
    (P, R, s, t, v), var = bcpd_align(
        trimmed_ref,
        trimmed_mov,
        outlier_prob=0.1,
        num_iter=20,
        tolerance=1e-6,
        kernel=gaussian_kernel,
        lambda_param=1.0,
        kernel_beta=0.5,
        gamma=1.0,
        kappa=math.inf,
        transform_mode="rigid",
        mask=trimmed_mask,
    )

    # Check that no NaN values appear in any output
    assert not jnp.any(jnp.isnan(R)), "Rotation matrix contains NaN"
    assert not jnp.isnan(s), "Scale factor is NaN"
    assert not jnp.any(jnp.isnan(t)), "Translation contains NaN"
    assert not jnp.any(jnp.isnan(v)), "Vector field contains NaN"
    assert not jnp.any(jnp.isnan(P)), "Matching matrix contains NaN"
    if isinstance(var, tuple):
        assert not jnp.isnan(var[0]), "Final variance is NaN"
    else:
        assert not jnp.any(jnp.isnan(var)), "Variance history contains NaN"

    # Verify trimmed dimensions
    assert trimmed_ref.shape[0] == 4
    assert trimmed_mov.shape[0] == 4
    assert trimmed_mask.shape == (4, 4)


def test_bcpd_trim_with_only_valid_matches():
    """When mask has no all-zero rows/cols, trimming should be a no-op."""
    ref = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mov = jnp.array([[0.1, 0.1], [1.1, 0.1], [0.1, 1.1]])
    mask = jnp.ones((3, 3))

    trimmed_ref, trimmed_mov, trimmed_mask, mov_idx, ref_idx = trim_with_mask(
        ref, mov, mask
    )

    (P, R, s, t, v), var = bcpd_align(
        trimmed_ref,
        trimmed_mov,
        outlier_prob=0.1,
        num_iter=10,
        tolerance=1e-6,
        kernel=gaussian_kernel,
        lambda_param=1.0,
        kernel_beta=0.5,
        gamma=1.0,
        kappa=math.inf,
        transform_mode="rigid",
        mask=trimmed_mask,
    )

    # No NaN
    assert not jnp.any(jnp.isnan(R))
    assert not jnp.isnan(s)
    assert not jnp.any(jnp.isnan(t))

    # Trimming should be a no-op
    assert jnp.allclose(trimmed_ref, ref)
    assert jnp.allclose(trimmed_mov, mov)
    assert jnp.allclose(trimmed_mask, mask)
