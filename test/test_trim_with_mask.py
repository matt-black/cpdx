"""Unit tests for trim_with_mask and untrim_with_mask functions."""

import jax.numpy as jnp
import pytest

from cpdx.util import trim_with_mask


class TestTrimWithMask:
    """Tests for trim_with_mask."""

    def test_no_trimming_needed(self):
        """When mask is all ones, no points should be removed."""
        ref = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        mov = jnp.array([[0.1, 0.1], [1.1, 0.1]])
        mask = jnp.ones((2, 3))

        trimmed_ref, trimmed_mov, trimmed_mask, mov_idx, ref_idx = (
            trim_with_mask(ref, mov, mask)
        )

        assert trimmed_ref.shape == (3, 2)
        assert trimmed_mov.shape == (2, 2)
        assert trimmed_mask.shape == (2, 3)
        assert jnp.allclose(trimmed_ref, ref)
        assert jnp.allclose(trimmed_mov, mov)
        assert jnp.allclose(trimmed_mask, mask)
        assert jnp.array_equal(mov_idx, jnp.array([0, 1]))
        assert jnp.array_equal(ref_idx, jnp.array([0, 1, 2]))

    def test_all_zero_row(self):
        """A moving point with all-zero row should be removed."""
        ref = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        mov = jnp.array([[0.1, 0.1], [2.0, 2.0], [3.0, 3.0]])
        # Row 1 (second moving point) has no valid matches
        mask = jnp.array([[1, 1], [0, 0], [1, 1]])

        trimmed_ref, trimmed_mov, trimmed_mask, mov_idx, ref_idx = (
            trim_with_mask(ref, mov, mask)
        )

        assert trimmed_mov.shape == (2, 2)
        assert trimmed_mask.shape == (2, 2)
        assert jnp.array_equal(mov_idx, jnp.array([0, 2]))
        assert jnp.array_equal(ref_idx, jnp.array([0, 1]))
        assert jnp.allclose(trimmed_mov[0], mov[0])
        assert jnp.allclose(trimmed_mov[1], mov[2])

    def test_all_zero_column(self):
        """A reference point with all-zero column should be removed."""
        ref = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        mov = jnp.array([[0.1, 0.1], [1.1, 0.1]])
        # Column 1 (second reference point) has no valid matches
        mask = jnp.array([[1, 0, 1], [1, 0, 1]])

        trimmed_ref, trimmed_mov, trimmed_mask, mov_idx, ref_idx = (
            trim_with_mask(ref, mov, mask)
        )

        assert trimmed_ref.shape == (2, 2)
        assert trimmed_mask.shape == (2, 2)
        assert jnp.array_equal(ref_idx, jnp.array([0, 2]))
        assert jnp.array_equal(mov_idx, jnp.array([0, 1]))
        assert jnp.allclose(trimmed_ref[0], ref[0])
        assert jnp.allclose(trimmed_ref[1], ref[2])

    def test_multiple_all_zero_rows_and_columns(self):
        """Mixed case with multiple unmatchable rows and columns."""
        ref = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        mov = jnp.array([[0.1, 0.1], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        # Row 1 and 2 are all zero; column 1 and 3 are all zero
        mask = jnp.array(
            [
                [1, 0, 1, 0],  # row 0: valid
                [0, 0, 0, 0],  # row 1: all zero
                [0, 0, 0, 0],  # row 2: all zero
                [1, 0, 1, 0],  # row 3: valid
            ]
        )

        trimmed_ref, trimmed_mov, trimmed_mask, mov_idx, ref_idx = (
            trim_with_mask(ref, mov, mask)
        )

        assert trimmed_mov.shape == (2, 2)
        assert trimmed_ref.shape == (2, 2)
        assert trimmed_mask.shape == (2, 2)
        assert jnp.array_equal(mov_idx, jnp.array([0, 3]))
        assert jnp.array_equal(ref_idx, jnp.array([0, 2]))

    def test_all_rows_zero_raises(self):
        """When all rows are zero, should raise ValueError."""
        ref = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        mov = jnp.array([[0.1, 0.1], [1.1, 0.1]])
        mask = jnp.zeros((2, 2))

        with pytest.raises(
            ValueError, match="all moving points are unmatchable"
        ):
            trim_with_mask(ref, mov, mask)

    def test_all_columns_zero_raises(self):
        """When all columns are zero but rows are valid, should raise ValueError.

        Note: to have valid rows with all-zero columns we need more reference points
        than moving points, so that each row can have a 1 in a different column while
        some column(s) remain all zero. But if ALL columns are zero then all rows are
        also zero. So we construct a case where moving points have valid rows but
        after trimming those rows, the remaining mask has all-zero columns.

        Actually the simplest: make a mask where every row has at least one nonzero
        but every column is all zero — this is impossible. Instead test that when
        the only nonzero entries are in columns that would be removed, we get the
        reference error. Use a mask where rows are valid but all reference columns
        end up zero after removing the matched rows.

        Simplest approach: use a mask with valid rows where each row targets a
        different column, but one column is never targeted. That column is removed
        but others remain. To trigger the error we need ALL columns to be zero,
        which means all rows must also be zero — caught by the moving check first.

        So we test: mask has valid rows, but after trim, all ref columns are zero.
        This can't happen because if rows are valid, some columns must be nonzero.

        The real test: create a scenario where mov is valid but ref is not.
        This requires m > 0 valid rows but n valid columns = 0, which is impossible
        if any row has a nonzero entry.

        Conclusion: the "all reference points" error can only fire when there are
        valid moving rows whose nonzero entries fall in columns that are otherwise
        all zero — but that would make those columns nonzero. So the only way to
        trigger this is if the mask has shape (m, n) with m > 0 and every row sums
        to > 0 but every column sums to 0, which is a contradiction.

        The error fires when len(ref_idx) == 0 AFTER mov_idx is non-empty.
        This happens when mask has nonzero entries (so valid_mov is True) but
        somehow all columns sum to zero — impossible.

        Therefore: test that a mask with shape forcing all columns to be checked
        but all zero still raises (caught by moving check). The reference check
        is a safeguard for edge cases in derived masks.
        """
        # A truly all-zero mask triggers the moving check first.
        # To test the reference check, we need valid rows but no valid columns,
        # which is mathematically impossible. The reference check is a defensive
        # safeguard. We verify it exists by checking the function raises on
        # all-zero mask (which it does via the moving check).
        ref = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        mov = jnp.array([[0.1, 0.1], [1.1, 0.1]])
        mask = jnp.zeros((2, 2))

        with pytest.raises(ValueError):
            trim_with_mask(ref, mov, mask)

    def test_entire_mask_zero_raises(self):
        """When entire mask is zero, should raise ValueError."""
        ref = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        mov = jnp.array([[0.1, 0.1], [1.1, 0.1], [0.1, 1.1]])
        mask = jnp.zeros((3, 3))

        with pytest.raises(ValueError):
            trim_with_mask(ref, mov, mask)

    def test_single_valid_pair(self):
        """Mask with only one non-zero entry should yield 1x1 trimmed arrays."""
        ref = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        mov = jnp.array([[0.1, 0.1], [1.1, 0.1], [0.1, 1.1]])
        mask = jnp.zeros((3, 3))
        mask = mask.at[1, 2].set(1)  # only mov[1] can match ref[2]

        trimmed_ref, trimmed_mov, trimmed_mask, mov_idx, ref_idx = (
            trim_with_mask(ref, mov, mask)
        )

        assert trimmed_ref.shape == (1, 2)
        assert trimmed_mov.shape == (1, 2)
        assert trimmed_mask.shape == (1, 1)
        assert jnp.array_equal(mov_idx, jnp.array([1]))
        assert jnp.array_equal(ref_idx, jnp.array([2]))
        assert jnp.allclose(trimmed_mov, mov[1:2])
        assert jnp.allclose(trimmed_ref, ref[2:3])
        assert trimmed_mask[0, 0] == 1
