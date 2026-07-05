"""Tests for point-wise uncertainty integration (Phases 1-3).

Phase 1: mahalanobis_dist in util.py
Phase 2: Unified expectation in _matching.py
Phase 3: maximization_uncertainty in rigid.py and affine.py

Strategy: Copy existing implementations as reference functions.
The new uncertainty-aware functions must match references when uncertainties are 0.
"""

import jax
import jax.numpy as jnp
import pytest

from cpdx._matching import expectation
from cpdx.affine import align as affine_align
from cpdx.affine import align_fixed_iter as affine_align_fixed_iter
from cpdx.affine import maximization as affine_maximization
from cpdx.affine import maximization_uncertainty as affine_maximization_uncertainty
from cpdx.rigid import align as rigid_align
from cpdx.rigid import align_fixed_iter as rigid_align_fixed_iter
from cpdx.rigid import maximization as rigid_maximization
from cpdx.rigid import maximization_uncertainty as rigid_maximization_uncertainty
from cpdx.util import mahalanobis_dist
from cpdx.util import sqdist


# ====================================================================
# Reference implementations (copies of original functions for testing)
# ====================================================================

def _ref_expectation(
    x, y_t, var, w
):
    """Reference: original expectation (no weights, no mask, no uncertainty)."""
    n, d = x.shape
    m, _ = y_t.shape
    d_t = sqdist(x, y_t).transpose()
    top = jnp.exp(jnp.negative(jnp.divide(d_t, 2 * var)))
    outl_term = jnp.divide(w, 1.0 - w) * jnp.divide(
        jnp.float_power(2 * jnp.pi * var, d / 2) * m, n
    )
    bot = jnp.add(
        jnp.clip(jnp.sum(top, axis=0, keepdims=True), jnp.finfo(x.dtype).eps),
        outl_term,
    )
    return jnp.divide(top, bot)


def _ref_expectation_weighted(
    x, y_t, var, w, alpha_m
):
    """Reference: expectation with mixing weights."""
    n, d = x.shape
    d_t = sqdist(x, y_t).transpose()
    top = alpha_m[:, None] * jnp.exp(jnp.negative(jnp.divide(d_t, 2 * var)))
    alpha_sum = jnp.sum(alpha_m)
    outl_term = jnp.divide(w, 1.0 - w) * jnp.divide(
        jnp.float_power(2 * jnp.pi * var, d / 2) * alpha_sum, n
    )
    bot = jnp.add(
        jnp.clip(jnp.sum(top, axis=0, keepdims=True), jnp.finfo(x.dtype).eps),
        outl_term,
    )
    return jnp.divide(top, bot)


def _ref_expectation_masked(
    x, y_t, var, w, mask
):
    """Reference: expectation with pairwise mask."""
    n, d = x.shape
    m, _ = y_t.shape
    d_t = sqdist(x, y_t).transpose()
    top = jnp.exp(jnp.negative(jnp.divide(d_t, 2 * var)))
    outl_term = jnp.divide(w, 1.0 - w) * jnp.divide(
        jnp.float_power(2 * jnp.pi * var, d / 2) * m, n
    )
    top = jnp.where(mask > 0, top, 0)
    n_mpr = jnp.sum(mask, axis=0, keepdims=True)
    n_msk = jnp.sum(mask)
    bot = jnp.add(
        jnp.clip(jnp.sum(top, axis=0, keepdims=True), jnp.finfo(x.dtype).eps),
        outl_term * (n_mpr / n_msk),
    )
    return jnp.divide(top, bot)


# ====================================================================
# Fixtures / test data
# ====================================================================

@pytest.fixture
def basic_points():
    """Small 2D point sets for deterministic testing."""
    x = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])  # (3, 2)
    y_t = jnp.array([[0.1, 0.1], [1.1, 0.9]])              # (2, 2)
    return x, y_t


@pytest.fixture
def zero_uncertainties(basic_points):
    """Zero uncertainty arrays matching basic_points shapes."""
    x, y_t = basic_points
    unc_x = jnp.zeros_like(x)
    unc_y = jnp.zeros_like(y_t)
    return x, y_t, unc_x, unc_y


# ====================================================================
# Phase 1: mahalanobis_dist tests
# ====================================================================

class TestMahalanobisDist:
    """Tests for the mahalanobis_dist function in util.py."""

    def test_shape(self, basic_points):
        """mahalanobis_dist returns correct shapes."""
        x, y_t = basic_points
        unc_x = jnp.ones_like(x) * 0.1
        unc_y = jnp.ones_like(y_t) * 0.1
        var = jnp.array(0.5)

        mahal, log_det = mahalanobis_dist(x, unc_x, y_t, unc_y, var)
        assert mahal.shape == (x.shape[0], y_t.shape[0])
        assert log_det.shape == (x.shape[0], y_t.shape[0])

    def test_symmetric_diagonal_zero(self, basic_points):
        """Mahalanobis distance is 0 when comparing a point to itself."""
        x, _ = basic_points
        unc_x = jnp.zeros_like(x)
        unc_y = jnp.zeros_like(x)
        var = jnp.array(0.5)

        mahal, _ = mahalanobis_dist(x, unc_x, x, unc_y, var)
        assert jnp.allclose(jnp.diag(mahal), jnp.zeros(x.shape[0]), atol=1e-6)

    def test_reduces_to_sqdist_zero_uncertainty(self, basic_points):
        """When uncertainties are zero, Mahalanobis distance = squared distance."""
        x, y_t = basic_points
        unc_x = jnp.zeros_like(x)
        unc_y = jnp.zeros_like(y_t)
        var = jnp.array(1.0)

        mahal, _ = mahalanobis_dist(x, unc_x, y_t, unc_y, var)
        sq = sqdist(x, y_t)
        assert jnp.allclose(mahal, sq, atol=1e-6)

    def test_log_det_matches_analytical(self, basic_points):
        """Log-determinant with zero uncertainty = d * log(var)."""
        x, y_t = basic_points
        d = x.shape[1]
        unc_x = jnp.zeros_like(x)
        unc_y = jnp.zeros_like(y_t)
        var = jnp.array(0.5)

        _, log_det = mahalanobis_dist(x, unc_x, y_t, unc_y, var)
        expected = d * jnp.log(var)
        assert jnp.allclose(log_det, expected, atol=1e-6)

    def test_nonzero_uncertainty_increases_distance_denominator(self):
        """Larger uncertainty should decrease Mahalanobis distance."""
        x = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        y = jnp.array([[0.5, 0.5]])
        var = jnp.array(0.1)

        unc_x_small = jnp.ones_like(x) * 0.01
        unc_y_small = jnp.ones_like(y) * 0.01
        unc_x_large = jnp.ones_like(x) * 1.0
        unc_y_large = jnp.ones_like(y) * 1.0

        mahal_small, _ = mahalanobis_dist(x, unc_x_small, y, unc_y_small, var)
        mahal_large, _ = mahalanobis_dist(x, unc_x_large, y, unc_y_large, var)

        # Larger uncertainty → smaller Mahalanobis distance
        assert jnp.all(mahal_large < mahal_small)


# ====================================================================
# Phase 2: Unified expectation tests
# ====================================================================

class TestUnifiedExpectation:
    """Tests for the unified expectation function in _matching.py."""

    def test_matches_reference_no_uncertainty(self, basic_points):
        """Unified expectation with no uncertainties matches original."""
        x, y_t = basic_points
        var = jnp.array(0.1)
        w = 0.1

        P_ref = _ref_expectation(x, y_t, var, w)
        P_new = expectation(x, y_t, var, w)

        assert jnp.allclose(P_ref, P_new, atol=1e-7)

    def test_matches_reference_weighted(self, basic_points):
        """Unified expectation with weights matches reference weighted."""
        x, y_t = basic_points
        var = jnp.array(0.1)
        w = 0.1
        alpha_m = jnp.array([2.0, 1.0])

        P_ref = _ref_expectation_weighted(x, y_t, var, w, alpha_m)
        P_new = expectation(x, y_t, var, w, moving_weights=alpha_m)

        assert jnp.allclose(P_ref, P_new, atol=1e-7)

    def test_matches_reference_masked(self, basic_points):
        """Unified expectation with mask matches reference masked."""
        x, y_t = basic_points
        var = jnp.array(0.1)
        w = 0.1
        mask = jnp.array([[1, 0, 1], [1, 1, 0]], dtype=float)

        P_ref = _ref_expectation_masked(x, y_t, var, w, mask)
        P_new = expectation(x, y_t, var, w, mask=mask)

        assert jnp.allclose(P_ref, P_new, atol=1e-7)

    def test_matches_reference_zero_uncertainty(self, zero_uncertainties):
        """Unified expectation with zero uncertainties matches original."""
        x, y_t, unc_x, unc_y = zero_uncertainties
        var = jnp.array(0.1)
        w = 0.1

        P_ref = _ref_expectation(x, y_t, var, w)
        P_new = expectation(x, y_t, var, w, unc_x=unc_x, unc_y=unc_y)

        assert jnp.allclose(P_ref, P_new, atol=1e-7)

    def test_zero_uncertainty_matches_reference_weighted(self, zero_uncertainties):
        """Unified expectation with zero uncertainties + weights matches ref."""
        x, y_t, unc_x, unc_y = zero_uncertainties
        var = jnp.array(0.1)
        w = 0.1
        alpha_m = jnp.array([1.5, 0.5])

        P_ref = _ref_expectation_weighted(x, y_t, var, w, alpha_m)
        P_new = expectation(x, y_t, var, w, moving_weights=alpha_m,
                           unc_x=unc_x, unc_y=unc_y)

        assert jnp.allclose(P_ref, P_new, atol=1e-7)

    def test_zero_uncertainty_matches_reference_masked(self, zero_uncertainties):
        """Unified expectation with zero uncertainties + mask matches ref."""
        x, y_t, unc_x, unc_y = zero_uncertainties
        var = jnp.array(0.1)
        w = 0.1
        mask = jnp.array([[1, 1, 0], [0, 1, 1]], dtype=float)

        P_ref = _ref_expectation_masked(x, y_t, var, w, mask)
        P_new = expectation(x, y_t, var, w, mask=mask,
                           unc_x=unc_x, unc_y=unc_y)

        assert jnp.allclose(P_ref, P_new, atol=1e-7)

    def test_columns_sum_le_one(self, basic_points):
        """Each column of P should sum to <= 1 (remaining mass = outliers)."""
        x, y_t = basic_points
        var = jnp.array(0.1)
        w = 0.1

        P = expectation(x, y_t, var, w)
        col_sums = jnp.sum(P, axis=0)
        assert jnp.all(col_sums <= 1.0 + 1e-6)

    def test_mask_enforcement(self, basic_points):
        """Masked entries should be exactly 0."""
        x, y_t = basic_points
        var = jnp.array(0.1)
        w = 0.1
        mask = jnp.array([[1, 0, 0], [0, 1, 1]], dtype=float)

        P = expectation(x, y_t, var, w, mask=mask)
        assert jnp.all(P[mask == 0] == 0)

    def test_nonzero_uncertainty_changes_result(self, zero_uncertainties):
        """Non-zero uncertainty should produce different result than zero."""
        x, y_t, unc_x, unc_y = zero_uncertainties
        var = jnp.array(0.1)
        w = 0.1

        P_zero = expectation(x, y_t, var, w, unc_x=unc_x, unc_y=unc_y)

        unc_x_big = jnp.ones_like(unc_x) * 0.5
        unc_y_big = jnp.ones_like(unc_y) * 0.5
        P_big = expectation(x, y_t, var, w, unc_x=unc_x_big, unc_y=unc_y_big)

        assert not jnp.allclose(P_zero, P_big, atol=1e-4)

    def test_nonzero_uncertainty_columns_sum_le_one(self, basic_points):
        """Column sums with uncertainty should still be <= 1."""
        x, y_t = basic_points
        var = jnp.array(0.1)
        w = 0.1
        unc_x = jnp.ones_like(x) * 0.1
        unc_y = jnp.ones_like(y_t) * 0.1

        P = expectation(x, y_t, var, w, unc_x=unc_x, unc_y=unc_y)
        col_sums = jnp.sum(P, axis=0)
        assert jnp.all(col_sums <= 1.0 + 1e-6)

    def test_combined_weights_mask_uncertainty(self, zero_uncertainties):
        """All optional params together should produce valid P."""
        x, y_t, unc_x, unc_y = zero_uncertainties
        var = jnp.array(0.1)
        w = 0.1
        alpha_m = jnp.array([1.5, 0.5])
        mask = jnp.array([[1, 1, 0], [0, 1, 1]], dtype=float)

        P = expectation(x, y_t, var, w,
                       moving_weights=alpha_m, mask=mask,
                       unc_x=unc_x, unc_y=unc_y)

        # Mask enforcement
        assert jnp.all(P[mask == 0] == 0)
        # Column sums
        col_sums = jnp.sum(P, axis=0)
        assert jnp.all(col_sums <= 1.0 + 1e-6)
        # Non-negative
        assert jnp.all(P >= 0)


# ====================================================================
# Phase 3: M-step (maximization_uncertainty) tests
# ====================================================================

class TestRigidMaximizationUncertainty:
    """Tests for maximization_uncertainty in rigid.py."""

    def test_zero_uncertainty_equivalence_transform(self, basic_points):
        """With zero uncertainty, transform params match base maximization."""
        x, y = basic_points
        var = jnp.array(0.1)
        w = 0.1
        tol = 1e-6
        P = expectation(x, y, var, w)

        (R_base, s_base, t_base), var_base = rigid_maximization(x, y, P, tol)
        unc_x = jnp.zeros_like(x)
        unc_y = jnp.zeros_like(y)
        (R_unc, s_unc, t_unc), var_unc = rigid_maximization_uncertainty(
            x, y, P, tol, var, unc_x, unc_y
        )

        assert jnp.allclose(R_base, R_unc, atol=1e-6)
        assert jnp.allclose(s_base, s_unc, atol=1e-6)
        assert jnp.allclose(t_base, t_unc, atol=1e-6)

    def test_zero_uncertainty_equivalence_variance(self, basic_points):
        """With zero uncertainty, variance matches base maximization."""
        x, y = basic_points
        var = jnp.array(0.1)
        w = 0.1
        tol = 1e-6
        P = expectation(x, y, var, w)

        _, var_base = rigid_maximization(x, y, P, tol)
        unc_x = jnp.zeros_like(x)
        unc_y = jnp.zeros_like(y)
        _, var_unc = rigid_maximization_uncertainty(
            x, y, P, tol, var, unc_x, unc_y
        )

        assert jnp.allclose(var_base, var_unc, atol=1e-6)

    def test_nonzero_uncertainty_changes_transform(self, basic_points):
        """Non-zero uncertainty should produce different transform."""
        x, y = basic_points
        var = jnp.array(0.1)
        w = 0.1
        tol = 1e-6
        P = expectation(x, y, var, w)

        (R_base, s_base, t_base), _ = rigid_maximization(x, y, P, tol)

        # Use non-uniform uncertainty to get a different effective weighting
        unc_x = jnp.array([[0.0, 0.5], [0.3, 0.1], [0.1, 0.4]])
        unc_y = jnp.array([[0.4, 0.2], [0.1, 0.3]])
        (R_unc, s_unc, t_unc), _ = rigid_maximization_uncertainty(
            x, y, P, tol, var, unc_x, unc_y
        )

        # At least one param should differ
        assert not (jnp.allclose(R_base, R_unc, atol=1e-4) and
                    jnp.allclose(s_base, s_unc, atol=1e-4) and
                    jnp.allclose(t_base, t_unc, atol=1e-4))

    def test_variance_non_negative(self, basic_points):
        """Variance from uncertainty-aware M-step should be non-negative."""
        x, y = basic_points
        var = jnp.array(0.1)
        w = 0.1
        tol = 1e-6
        P = expectation(x, y, var, w)

        unc_x = jnp.ones_like(x) * 0.05
        unc_y = jnp.ones_like(y) * 0.05
        _, new_var = rigid_maximization_uncertainty(
            x, y, P, tol, var, unc_x, unc_y
        )
        assert new_var >= 0

    def test_zero_uncertainty_equiv_m_not_eq_d(self):
        """Zero-unc equivalence with m != d to catch broadcasting bugs."""
        x = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])  # (n=3, d=2)
        y = jnp.array([[0.1, 0.1], [1.1, 0.9], [2.5, 0.7], [0.8, 1.2]])  # (m=4, d=2)
        var = jnp.array(0.1)
        w = 0.1
        tol = 1e-6
        P = expectation(x, y, var, w)  # (m=4, n=3)

        (R_base, s_base, t_base), var_base = rigid_maximization(x, y, P, tol)
        unc_x = jnp.zeros_like(x)
        unc_y = jnp.zeros_like(y)
        (R_unc, s_unc, t_unc), var_unc = rigid_maximization_uncertainty(
            x, y, P, tol, var, unc_x, unc_y
        )

        assert jnp.allclose(R_base, R_unc, atol=1e-5)
        assert jnp.allclose(s_base, s_unc, atol=1e-5)
        assert jnp.allclose(t_base, t_unc, atol=1e-5)
        assert jnp.allclose(var_base, var_unc, atol=1e-5)

    def test_zero_uncertainty_equiv_3d(self):
        """Zero-unc equivalence in 3D."""
        x = jnp.array([
            [0.0, 0.0, 0.0], [1.0, 1.0, 0.5], [2.0, 0.5, 1.0],
            [3.0, 0.3, 0.2], [0.5, 2.0, 1.5],
        ])  # (n=5, d=3)
        y = jnp.array([
            [0.1, 0.1, 0.05], [1.1, 0.9, 0.6],
            [2.5, 0.7, 0.9], [0.8, 1.2, 0.3],
        ])  # (m=4, d=3)
        var = jnp.array(0.1)
        w = 0.1
        tol = 1e-6
        P = expectation(x, y, var, w)  # (m=4, n=5)

        (R_base, s_base, t_base), var_base = rigid_maximization(x, y, P, tol)
        unc_x = jnp.zeros_like(x)
        unc_y = jnp.zeros_like(y)
        (R_unc, s_unc, t_unc), var_unc = rigid_maximization_uncertainty(
            x, y, P, tol, var, unc_x, unc_y
        )

        assert jnp.allclose(R_base, R_unc, atol=1e-5)
        assert jnp.allclose(s_base, s_unc, atol=1e-5)
        assert jnp.allclose(t_base, t_unc, atol=1e-5)
        assert jnp.allclose(var_base, var_unc, atol=1e-5)


class TestAffineMaximizationUncertainty:
    """Tests for maximization_uncertainty in affine.py."""

    def test_zero_uncertainty_equivalence_transform(self):
        """With zero uncertainty, transform params match base maximization."""
        # Use m >= 3 to avoid near-singular ypy matrix (m=2, d=2 is ill-posed)
        x = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 0.3]])  # (n=4, d=2)
        y = jnp.array([[0.1, 0.1], [1.1, 0.9], [2.5, 0.7], [0.8, 1.2]])  # (m=4, d=2)
        var = jnp.array(0.1)
        w = 0.1
        tol = 1e-6
        P = expectation(x, y, var, w)

        (A_base, t_base), var_base = affine_maximization(x, y, P, tol)
        unc_x = jnp.zeros_like(x)
        unc_y = jnp.zeros_like(y)
        (A_unc, t_unc), var_unc = affine_maximization_uncertainty(
            x, y, P, tol, var, unc_x, unc_y
        )

        assert jnp.allclose(A_base, A_unc, atol=1e-5)
        assert jnp.allclose(t_base, t_unc, atol=1e-5)

    def test_zero_uncertainty_equivalence_variance(self):
        """With zero uncertainty, variance matches base maximization."""
        x = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 0.3]])  # (n=4, d=2)
        y = jnp.array([[0.1, 0.1], [1.1, 0.9], [2.5, 0.7], [0.8, 1.2]])  # (m=4, d=2)
        var = jnp.array(0.1)
        w = 0.1
        tol = 1e-6
        P = expectation(x, y, var, w)

        _, var_base = affine_maximization(x, y, P, tol)
        unc_x = jnp.zeros_like(x)
        unc_y = jnp.zeros_like(y)
        _, var_unc = affine_maximization_uncertainty(
            x, y, P, tol, var, unc_x, unc_y
        )

        assert jnp.allclose(var_base, var_unc, atol=1e-5)

    def test_zero_uncertainty_equiv_m_not_eq_d(self):
        """Zero-unc equivalence with m != d to catch broadcasting bugs."""
        x = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 0.3]])  # (n=4, d=2)
        y = jnp.array([[0.1, 0.1], [1.1, 0.9], [2.5, 0.7], [0.8, 1.2]])  # (m=4, d=2)
        # Use different m than d to catch shape bugs
        x2 = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])             # (n=3, d=2)
        y2 = jnp.array([[0.1, 0.1], [1.1, 0.9], [2.5, 0.7], [0.8, 1.2]])  # (m=4, d=2)
        var = jnp.array(0.1)
        w = 0.1
        tol = 1e-6
        P = expectation(x2, y2, var, w)  # (m=4, n=3)

        (A_base, t_base), var_base = affine_maximization(x2, y2, P, tol)
        unc_x = jnp.zeros_like(x2)
        unc_y = jnp.zeros_like(y2)
        (A_unc, t_unc), var_unc = affine_maximization_uncertainty(
            x2, y2, P, tol, var, unc_x, unc_y
        )

        assert jnp.allclose(A_base, A_unc, atol=1e-5)
        assert jnp.allclose(t_base, t_unc, atol=1e-5)
        assert jnp.allclose(var_base, var_unc, atol=1e-5)

    def test_zero_uncertainty_equiv_3d(self):
        """Zero-unc equivalence in 3D."""
        x = jnp.array([
            [0.0, 0.0, 0.0], [1.0, 1.0, 0.5], [2.0, 0.5, 1.0],
            [3.0, 0.3, 0.2], [0.5, 2.0, 1.5],
        ])  # (n=5, d=3)
        y = jnp.array([
            [0.1, 0.1, 0.05], [1.1, 0.9, 0.6],
            [2.5, 0.7, 0.9], [0.8, 1.2, 0.3],
        ])  # (m=4, d=3)
        var = jnp.array(0.1)
        w = 0.1
        tol = 1e-6
        P = expectation(x, y, var, w)  # (m=4, n=5)

        (A_base, t_base), var_base = affine_maximization(x, y, P, tol)
        unc_x = jnp.zeros_like(x)
        unc_y = jnp.zeros_like(y)
        (A_unc, t_unc), var_unc = affine_maximization_uncertainty(
            x, y, P, tol, var, unc_x, unc_y
        )

        assert jnp.allclose(A_base, A_unc, atol=1e-5)
        assert jnp.allclose(t_base, t_unc, atol=1e-5)
        assert jnp.allclose(var_base, var_unc, atol=1e-5)

    def test_nonzero_uncertainty_changes_transform(self, basic_points):
        """Non-zero uncertainty should produce different transform."""
        x, y = basic_points
        var = jnp.array(0.1)
        w = 0.1
        tol = 1e-6
        P = expectation(x, y, var, w)

        (A_base, t_base), _ = affine_maximization(x, y, P, tol)

        unc_x = jnp.ones_like(x) * 0.3
        unc_y = jnp.ones_like(y) * 0.3
        (A_unc, t_unc), _ = affine_maximization_uncertainty(
            x, y, P, tol, var, unc_x, unc_y
        )

        assert not (jnp.allclose(A_base, A_unc, atol=1e-4) and
                    jnp.allclose(t_base, t_unc, atol=1e-4))

    def test_nonzero_uncertainty_non_uniform(self):
        """Non-uniform uncertainty should produce different transform per dim."""
        x = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 0.3]])
        y = jnp.array([[0.1, 0.1], [1.1, 0.9], [2.5, 0.7], [0.8, 1.2]])
        var = jnp.array(0.1)
        w = 0.1
        tol = 1e-6
        P = expectation(x, y, var, w)

        (A_base, t_base), _ = affine_maximization(x, y, P, tol)

        # Very different uncertainty per dimension
        unc_x = jnp.array([[0.1, 0.0], [0.0, 0.2], [0.3, 0.1], [0.05, 0.15]])
        unc_y = jnp.array([[0.15, 0.1], [0.0, 0.25], [0.2, 0.05], [0.1, 0.1]])
        (A_unc, t_unc), _ = affine_maximization_uncertainty(
            x, y, P, tol, var, unc_x, unc_y
        )

        assert not (jnp.allclose(A_base, A_unc, atol=1e-4) and
                    jnp.allclose(t_base, t_unc, atol=1e-4))

    def test_variance_non_negative(self, basic_points):
        """Variance from uncertainty-aware M-step should be non-negative."""
        x, y = basic_points
        var = jnp.array(0.1)
        w = 0.1
        tol = 1e-6
        P = expectation(x, y, var, w)

        unc_x = jnp.ones_like(x) * 0.05
        unc_y = jnp.ones_like(y) * 0.05
        _, new_var = affine_maximization_uncertainty(
            x, y, P, tol, var, unc_x, unc_y
        )
        assert new_var >= 0


# ====================================================================
# Phase 3: Full alignment regression tests
# ====================================================================

class TestRigidAlignWithUncertainty:
    """Tests for rigid align/align_fixed_iter with uncertainty params."""

    def test_align_unchanged_without_uncertainty(self, basic_points):
        """align without uncertainty produces identical results to baseline."""
        ref, mov = basic_points
        w = 0.1
        max_iter = 20
        tol = 1e-6

        result_base = rigid_align(ref, mov, w, max_iter, tol)
        result_unc = rigid_align(ref, mov, w, max_iter, tol,
                                 unc_ref=None, unc_mov=None)

        (_, R_base, s_base, t_base), (var_base, iters_base) = result_base
        (_, R_unc, s_unc, t_unc), (var_unc, iters_unc) = result_unc

        assert jnp.allclose(R_base, R_unc, atol=1e-6)
        assert jnp.allclose(s_base, s_unc, atol=1e-6)
        assert jnp.allclose(t_base, t_unc, atol=1e-6)
        assert jnp.allclose(var_base, var_unc, atol=1e-6)
        assert iters_base == iters_unc

    def test_align_fixed_iter_unchanged_without_uncertainty(self, basic_points):
        """align_fixed_iter without uncertainty matches baseline."""
        ref, mov = basic_points
        w = 0.1
        num_iter = 10

        (_, R_base, s_base, t_base), varz_base = rigid_align_fixed_iter(
            ref, mov, w, num_iter
        )
        (_, R_unc, s_unc, t_unc), varz_unc = rigid_align_fixed_iter(
            ref, mov, w, num_iter, unc_ref=None, unc_mov=None
        )

        assert jnp.allclose(R_base, R_unc, atol=1e-6)
        assert jnp.allclose(s_base, s_unc, atol=1e-6)
        assert jnp.allclose(t_base, t_unc, atol=1e-6)
        assert jnp.allclose(varz_base, varz_unc, atol=1e-6)

    def test_align_with_uncertainty_produces_valid_result(self, basic_points):
        """align with uncertainty converges to valid transform."""
        ref, mov = basic_points
        unc_ref = jnp.ones_like(ref) * 0.01
        unc_mov = jnp.ones_like(mov) * 0.01

        (_, R, s, t), (var, iters) = rigid_align(
            ref, mov, 0.1, 20, 1e-6,
            unc_ref=unc_ref, unc_mov=unc_mov,
        )

        # R should be a valid rotation matrix
        assert R.shape == (2, 2)
        assert jnp.allclose(R @ R.T, jnp.eye(2), atol=1e-4)
        assert jnp.abs(jnp.linalg.det(R) - 1.0) < 1e-4
        assert var >= 0
        assert iters > 0


class TestAffineAlignWithUncertainty:
    """Tests for affine align/align_fixed_iter with uncertainty params."""

    def test_align_unchanged_without_uncertainty(self, basic_points):
        """align without uncertainty produces identical results to baseline."""
        ref, mov = basic_points
        w = 0.1
        max_iter = 20
        tol = 1e-6

        result_base = affine_align(ref, mov, w, max_iter, tol)
        result_unc = affine_align(ref, mov, w, max_iter, tol,
                                  unc_ref=None, unc_mov=None)

        (_, A_base, t_base), (var_base, iters_base) = result_base
        (_, A_unc, t_unc), (var_unc, iters_unc) = result_unc

        assert jnp.allclose(A_base, A_unc, atol=1e-6)
        assert jnp.allclose(t_base, t_unc, atol=1e-6)
        assert jnp.allclose(var_base, var_unc, atol=1e-6)
        assert iters_base == iters_unc

    def test_align_fixed_iter_unchanged_without_uncertainty(self, basic_points):
        """align_fixed_iter without uncertainty matches baseline."""
        ref, mov = basic_points
        w = 0.1
        num_iter = 10

        (_, A_base, t_base), varz_base = affine_align_fixed_iter(
            ref, mov, w, num_iter
        )
        (_, A_unc, t_unc), varz_unc = affine_align_fixed_iter(
            ref, mov, w, num_iter, unc_ref=None, unc_mov=None
        )

        assert jnp.allclose(A_base, A_unc, atol=1e-6)
        assert jnp.allclose(t_base, t_unc, atol=1e-6)
        assert jnp.allclose(varz_base, varz_unc, atol=1e-6)

    def test_align_with_uncertainty_produces_valid_result(self, basic_points):
        """align with uncertainty converges to valid transform."""
        ref, mov = basic_points
        unc_ref = jnp.ones_like(ref) * 0.01
        unc_mov = jnp.ones_like(mov) * 0.01

        (_, A, t), (var, iters) = affine_align(
            ref, mov, 0.1, 20, 1e-6,
            unc_ref=unc_ref, unc_mov=unc_mov,
        )

        assert A.shape == (2, 2)
        assert t.shape == (2,)
        assert var >= 0
        assert iters > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ====================================================================
# Additional robustness and end-to-end recovery tests
# ====================================================================

class TestExtremeUncertainty:
    """Tests for extreme uncertainty regimes."""

    def test_extreme_uniformity_e_step(self, basic_points):
        """When uncertainty dominates, P should approach uniform probabilities."""
        x, y_t = basic_points
        n, d = x.shape
        m, _ = y_t.shape
        var = jnp.array(0.1)
        w = 0.1

        # Very large uncertainty — should make all pairs roughly equally likely
        unc_x = jnp.ones_like(x) * 1000.0
        unc_y = jnp.ones_like(y_t) * 1000.0

        P = expectation(x, y_t, var, w, unc_x=unc_x, unc_y=unc_y)

        # Each column should have roughly equal mass across rows
        # (allowing for outlier mass). Check coefficient of variation per column.
        col_means = jnp.mean(P, axis=0)  # mean across rows for each column
        col_stds = jnp.std(P, axis=0)
        # CV should be small (near-uniform)
        cv = col_stds / (col_means + 1e-12)
        assert jnp.all(cv < 0.1), f"Columns not uniform enough: cv={cv}"

    def test_general_path_column_stability_large_uncertainty(self):
        """Column sums ≤ 1 with large, non-uniform uncertainty."""
        x = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 0.3]])
        y_t = jnp.array([[0.1, 0.1], [1.1, 0.9], [2.5, 0.7]])
        var = jnp.array(0.1)
        w = 0.1

        # Large, non-uniform uncertainty
        unc_x = jnp.array([[0.5, 50.0], [10.0, 0.3], [50.0, 1.0], [0.1, 0.5]])
        unc_y = jnp.array([[0.2, 100.0], [10.0, 0.1], [50.0, 2.0]])

        P = expectation(x, y_t, var, w, unc_x=unc_x, unc_y=unc_y)
        col_sums = jnp.sum(P, axis=0)
        assert jnp.all(col_sums <= 1.0 + 1e-5)

    def test_negative_residual_floor_rigid(self):
        """When trace sum > residuals, variance falls to tolerance floor."""
        x = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
        y = jnp.array([[0.0, 0.0], [1.0, 1.0]])  # nearly identical to subset of x
        var = jnp.array(0.1)
        w = 0.1
        tol = 1e-6
        P = expectation(x, y, var, w)

        # Uncertainty so large it exceeds all residuals
        unc_x = jnp.ones_like(x) * 100.0
        unc_y = jnp.ones_like(y) * 100.0
        _, new_var = rigid_maximization_uncertainty(
            x, y, P, tol, var, unc_x, unc_y
        )
        # Should be clamped to tolerance / 10
        assert new_var > 0
        assert new_var <= tol

    def test_negative_residual_floor_affine(self):
        """When trace sum > residuals, variance falls to tolerance floor (affine)."""
        x = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 0.3]])
        y = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 0.3]])  # identical
        var = jnp.array(0.1)
        w = 0.1
        tol = 1e-6
        P = expectation(x, y, var, w)

        # Uncertainty so large it exceeds all residuals
        unc_x = jnp.ones_like(x) * 100.0
        unc_y = jnp.ones_like(y) * 100.0
        _, new_var = affine_maximization_uncertainty(
            x, y, P, tol, var, unc_x, unc_y
        )
        # Should be clamped to tolerance / 10
        assert new_var > 0
        assert new_var <= tol


class TestVarianceDecreasesWithHigherUncertainty:
    """Higher uncertainty → lower estimated σ² (more explained by measurement error)."""

    def test_variance_decreases_rigid(self):
        x = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 0.3]])
        y = jnp.array([[0.1, 0.1], [1.1, 0.9], [2.5, 0.7], [0.8, 1.2]])
        var = jnp.array(0.5)
        w = 0.1
        tol = 1e-6
        P = expectation(x, y, var, w)

        # Low uncertainty
        unc_x_low = jnp.ones_like(x) * 0.001
        unc_y_low = jnp.ones_like(y) * 0.001
        _, var_low = rigid_maximization_uncertainty(
            x, y, P, tol, var, unc_x_low, unc_y_low
        )

        # High uncertainty
        unc_x_high = jnp.ones_like(x) * 0.1
        unc_y_high = jnp.ones_like(y) * 0.1
        _, var_high = rigid_maximization_uncertainty(
            x, y, P, tol, var, unc_x_high, unc_y_high
        )

        assert var_high < var_low

    def test_variance_decreases_affine(self):
        x = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 0.3]])
        y = jnp.array([[0.1, 0.1], [1.1, 0.9], [2.5, 0.7], [0.8, 1.2]])
        var = jnp.array(0.5)
        w = 0.1
        tol = 1e-6
        P = expectation(x, y, var, w)

        # Low uncertainty
        unc_x_low = jnp.ones_like(x) * 0.001
        unc_y_low = jnp.ones_like(y) * 0.001
        _, var_low = affine_maximization_uncertainty(
            x, y, P, tol, var, unc_x_low, unc_y_low
        )

        # High uncertainty
        unc_x_high = jnp.ones_like(x) * 0.1
        unc_y_high = jnp.ones_like(y) * 0.1
        _, var_high = affine_maximization_uncertainty(
            x, y, P, tol, var, unc_x_high, unc_y_high
        )

        assert var_high < var_low


class TestRigidTransformRecovery:
    """Recovery of known rigid transforms with uncertainty — full alignment."""

    def _generate_points(self, key, n=100):
        """Generate random 2D points centered near origin."""
        pts = jax.random.uniform(key, (n, 2), minval=-5.0, maxval=5.0)
        return pts

    def test_recover_rotation_scale_translation(self):
        """Recover combined rotation + scale + translation."""
        from cpdx.rigid import transform as rigid_transform

        key = jax.random.PRNGKey(42)
        mov = self._generate_points(key)

        # Ground truth: 30° rotation, scale 1.5, translation (2, -1)
        angle = jnp.pi / 6
        R_true = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
        ])
        s_true = jnp.array(1.5)
        t_true = jnp.array([2.0, -1.0])

        # Apply transform: ref = s * (mov @ R.T) + t
        ref = rigid_transform(mov, R_true, s_true, t_true)

        # Small uncertainty
        unc_ref = jnp.ones_like(ref) * 0.01
        unc_mov = jnp.ones_like(mov) * 0.01

        (_, R, s, t), (var, iters) = rigid_align(
            ref, mov, 0.01, 100, 1e-8,
            unc_ref=unc_ref, unc_mov=unc_mov,
        )

        assert var >= 0
        assert iters > 0
        # Check rotation (up to sign — but CPD should find correct one)
        assert jnp.allclose(R, R_true, atol=0.05)
        assert jnp.allclose(s, s_true, atol=0.05)
        assert jnp.allclose(t, t_true, atol=0.1)

    def test_recover_pure_rotation(self):
        """Recover pure rotation (scale=1, translation=0)."""
        from cpdx.rigid import transform as rigid_transform

        key = jax.random.PRNGKey(123)
        mov = self._generate_points(key)

        angle = -jnp.pi / 4  # -45°
        R_true = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
        ])
        s_true = jnp.array(1.0)
        t_true = jnp.zeros(2)

        ref = rigid_transform(mov, R_true, s_true, t_true)

        unc_ref = jnp.ones_like(ref) * 0.01
        unc_mov = jnp.ones_like(mov) * 0.01

        (_, R, s, t), (var, _) = rigid_align(
            ref, mov, 0.01, 100, 1e-8,
            unc_ref=unc_ref, unc_mov=unc_mov,
        )

        assert jnp.allclose(R, R_true, atol=0.05)
        assert jnp.allclose(s, s_true, atol=0.05)
        assert jnp.allclose(t, t_true, atol=0.1)

    def test_recover_pure_translation(self):
        """Recover pure translation (scale=1, identity rotation)."""
        from cpdx.rigid import transform as rigid_transform

        key = jax.random.PRNGKey(456)
        mov = self._generate_points(key)

        R_true = jnp.eye(2)
        s_true = jnp.array(1.0)
        t_true = jnp.array([-3.0, 4.0])

        ref = rigid_transform(mov, R_true, s_true, t_true)

        unc_ref = jnp.ones_like(ref) * 0.01
        unc_mov = jnp.ones_like(mov) * 0.01

        (_, R, s, t), (var, _) = rigid_align(
            ref, mov, 0.01, 100, 1e-8,
            unc_ref=unc_ref, unc_mov=unc_mov,
        )

        assert jnp.allclose(R, R_true, atol=0.05)
        assert jnp.allclose(s, s_true, atol=0.05)
        assert jnp.allclose(t, t_true, atol=0.1)


class TestAffineTransformRecovery:
    """Recovery of known affine transforms with uncertainty — full alignment."""

    def _generate_points(self, key, n=100):
        """Generate random 2D points centered near origin."""
        pts = jax.random.uniform(key, (n, 2), minval=-5.0, maxval=5.0)
        return pts

    def test_recover_scaling_shear(self):
        """Recover non-uniform scaling + shear."""
        from cpdx.affine import transform as affine_transform

        key = jax.random.PRNGKey(789)
        mov = self._generate_points(key)

        # Non-uniform scaling + shear
        A_true = jnp.array([[1.5, 0.3], [0.2, 0.8]])
        t_true = jnp.array([1.0, -0.5])

        ref = affine_transform(mov, A_true, t_true)

        unc_ref = jnp.ones_like(ref) * 0.01
        unc_mov = jnp.ones_like(mov) * 0.01

        (_, A, t), (var, _) = affine_align(
            ref, mov, 0.01, 100, 1e-8,
            unc_ref=unc_ref, unc_mov=unc_mov,
        )

        assert jnp.allclose(A, A_true, atol=0.1)
        assert jnp.allclose(t, t_true, atol=0.1)

    def test_recover_general_affine(self):
        """Recover general affine with rotation-like component."""
        from cpdx.affine import transform as affine_transform

        key = jax.random.PRNGKey(101112)
        mov = self._generate_points(key)

        # General affine: rotation + scale + shear
        angle = jnp.pi / 8
        A_true = jnp.array([
            [1.2 * jnp.cos(angle), 1.2 * jnp.sin(angle) + 0.2],
            [-0.8 * jnp.sin(angle), 0.8 * jnp.cos(angle)],
        ])
        t_true = jnp.array([0.5, 1.5])

        ref = affine_transform(mov, A_true, t_true)

        unc_ref = jnp.ones_like(ref) * 0.01
        unc_mov = jnp.ones_like(mov) * 0.01

        (_, A, t), (var, _) = affine_align(
            ref, mov, 0.01, 100, 1e-8,
            unc_ref=unc_ref, unc_mov=unc_mov,
        )

        assert jnp.allclose(A, A_true, atol=0.1)
        assert jnp.allclose(t, t_true, atol=0.15)


# ====================================================================
# Phase 4: Nonrigid uncertainty tests
# ====================================================================

from cpdx.nonrigid import align as nonrigid_align
from cpdx.nonrigid import align_fixed_iter as nonrigid_align_fixed_iter
from cpdx.nonrigid import maximization as nonrigid_maximization
from cpdx.nonrigid import maximization_uncertainty as nonrigid_maximization_uncertainty


class TestNonrigidMaximizationUncertainty:
    """Tests for maximization_uncertainty in nonrigid.py."""

    def _setup(self):
        """Create small test data with kernel matrix."""
        x = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 0.3]])
        y = jnp.array([[0.1, 0.1], [1.1, 0.9], [2.5, 0.7], [0.8, 1.2]])
        var = jnp.array(0.1)
        G = jnp.exp(-sqdist(y, y) / (2 * 0.5))
        w = 0.1
        P = expectation(x, y, var, w)
        return x, y, P, G, var

    def test_zero_uncertainty_equivalence_W(self):
        """With zero uncertainty, W matches base maximization."""
        x, y, P, G, var = self._setup()
        tol = 1e-6
        reg = 1.0

        W_base, var_base = nonrigid_maximization(x, y, P, G, var, reg, tol)
        unc_x = jnp.zeros_like(x)
        unc_y = jnp.zeros_like(y)
        W_unc, var_unc = nonrigid_maximization_uncertainty(
            x, y, P, G, var, reg, tol, unc_x, unc_y
        )

        assert jnp.allclose(W_base, W_unc, atol=1e-5)

    def test_zero_uncertainty_equivalence_variance(self):
        """With zero uncertainty, variance matches base maximization."""
        x, y, P, G, var = self._setup()
        tol = 1e-6
        reg = 1.0

        _, var_base = nonrigid_maximization(x, y, P, G, var, reg, tol)
        unc_x = jnp.zeros_like(x)
        unc_y = jnp.zeros_like(y)
        _, var_unc = nonrigid_maximization_uncertainty(
            x, y, P, G, var, reg, tol, unc_x, unc_y
        )

        assert jnp.allclose(var_base, var_unc, atol=1e-5)

    def test_nonzero_uncertainty_changes_W(self):
        """Non-zero uncertainty should produce different coefficients."""
        x, y, P, G, var = self._setup()
        tol = 1e-6
        reg = 1.0

        W_base, _ = nonrigid_maximization(x, y, P, G, var, reg, tol)

        unc_x = jnp.ones_like(x) * 0.3
        unc_y = jnp.ones_like(y) * 0.3
        W_unc, _ = nonrigid_maximization_uncertainty(
            x, y, P, G, var, reg, tol, unc_x, unc_y
        )

        assert not jnp.allclose(W_base, W_unc, atol=1e-4)

    def test_variance_non_negative(self):
        """Variance from uncertainty-aware M-step should be non-negative."""
        x, y, P, G, var = self._setup()
        tol = 1e-6
        reg = 1.0

        unc_x = jnp.ones_like(x) * 0.05
        unc_y = jnp.ones_like(y) * 0.05
        _, new_var = nonrigid_maximization_uncertainty(
            x, y, P, G, var, reg, tol, unc_x, unc_y
        )
        assert new_var >= 0


class TestNonrigidAlignWithUncertainty:
    """Tests for nonrigid align/align_fixed_iter with uncertainty params."""

    def _setup_points(self):
        ref = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5], [3.0, 0.3]])
        mov = jnp.array([[0.1, 0.1], [1.1, 0.9], [2.5, 0.7], [0.8, 1.2]])
        return ref, mov

    def test_align_unchanged_without_uncertainty(self):
        """align without uncertainty produces identical results to baseline."""
        ref, mov = self._setup_points()
        w = 0.1
        reg = 1.0
        kvar = 0.5
        max_iter = 20
        tol = 1e-6

        result_base = nonrigid_align(ref, mov, w, reg, kvar, max_iter, tol)
        result_unc = nonrigid_align(ref, mov, w, reg, kvar, max_iter, tol,
                                    unc_ref=None, unc_mov=None)

        (_, _, W_base), (var_base, iters_base) = result_base
        (_, _, W_unc), (var_unc, iters_unc) = result_unc

        assert jnp.allclose(W_base, W_unc, atol=1e-6)
        assert jnp.allclose(var_base, var_unc, atol=1e-6)
        assert iters_base == iters_unc

    def test_align_fixed_iter_unchanged_without_uncertainty(self):
        """align_fixed_iter without uncertainty matches baseline."""
        ref, mov = self._setup_points()
        w = 0.1
        reg = 1.0
        kvar = 0.5
        num_iter = 10

        (_, _, W_base), varz_base = nonrigid_align_fixed_iter(
            ref, mov, w, reg, kvar, num_iter
        )
        (_, _, W_unc), varz_unc = nonrigid_align_fixed_iter(
            ref, mov, w, reg, kvar, num_iter, unc_ref=None, unc_mov=None
        )

        assert jnp.allclose(W_base, W_unc, atol=1e-6)
        assert jnp.allclose(varz_base, varz_unc, atol=1e-6)

    def test_align_with_uncertainty_produces_valid_result(self):
        """align with uncertainty converges to valid result."""
        ref, mov = self._setup_points()
        unc_ref = jnp.ones_like(ref) * 0.01
        unc_mov = jnp.ones_like(mov) * 0.01

        (_, _, W), (var, iters) = nonrigid_align(
            ref, mov, 0.1, 1.0, 0.5, 20, 1e-6,
            unc_ref=unc_ref, unc_mov=unc_mov,
        )

        assert W.shape == mov.shape
        assert var >= 0
        assert iters > 0


class TestNonrigidTransformRecovery:
    """Recovery of known nonrigid deformations with uncertainty — full alignment."""

    def _generate_points(self, key, n=100):
        """Generate random 2D points centered near origin."""
        pts = jax.random.uniform(key, (n, 2), minval=-5.0, maxval=5.0)
        return pts

    def test_recover_uniform_displacement(self):
        """Recover small uniform displacement (translation-like deformation)."""
        key = jax.random.PRNGKey(2001)
        mov = self._generate_points(key)

        # Uniform displacement
        v = jnp.array([0.3, -0.2])
        ref = mov + v

        unc_ref = jnp.ones_like(ref) * 0.01
        unc_mov = jnp.ones_like(mov) * 0.01

        (_, G, W), (var, _) = nonrigid_align(
            ref, mov, 0.01, 1.0, 0.5, 100, 1e-8,
            unc_ref=unc_ref, unc_mov=unc_mov,
        )

        # Verify transformed points match reference
        ref_recovered = mov + G @ W
        assert jnp.allclose(ref_recovered, ref, atol=0.1)

    def test_recover_spatially_varying_displacement(self):
        """Recover smooth position-dependent displacement (sin/cos)."""
        key = jax.random.PRNGKey(2002)
        mov = self._generate_points(key)

        # Smooth spatially-varying displacement
        v = jnp.stack([
            0.2 * jnp.sin(mov[:, 0]),
            0.2 * jnp.cos(mov[:, 1]),
        ], axis=1)
        ref = mov + v

        unc_ref = jnp.ones_like(ref) * 0.01
        unc_mov = jnp.ones_like(mov) * 0.01

        (_, G, W), (var, _) = nonrigid_align(
            ref, mov, 0.01, 1.0, 0.5, 100, 1e-8,
            unc_ref=unc_ref, unc_mov=unc_mov,
        )

        ref_recovered = mov + G @ W
        assert jnp.allclose(ref_recovered, ref, atol=0.15)

    def test_recover_near_identity(self):
        """When points are already aligned, transformed points should match reference."""
        key = jax.random.PRNGKey(2003)
        mov = self._generate_points(key)
        ref = mov  # No deformation needed

        unc_ref = jnp.ones_like(ref) * 0.01
        unc_mov = jnp.ones_like(mov) * 0.01

        (_, G, W), (var, _) = nonrigid_align(
            ref, mov, 0.01, 1.0, 0.5, 100, 1e-8,
            unc_ref=unc_ref, unc_mov=unc_mov,
        )

        # Transformed points should be close to original (reference)
        ref_recovered = mov + G @ W
        assert jnp.allclose(ref_recovered, ref, atol=0.15)

    def test_recover_synthetic_W(self):
        """Recover synthetic coefficient matrix W via deformation field."""
        key = jax.random.PRNGKey(2004)
        mov = self._generate_points(key)

        # Compute kernel matrix
        kvar = 0.5
        G = jnp.exp(-sqdist(mov, mov) / (2 * kvar))

        # Generate synthetic W — small random coefficients
        W_true = 0.1 * jax.random.uniform(key, (mov.shape[0], 2), minval=-1, maxval=1)

        # Ground truth: ref = mov + G @ W_true
        ref = mov + G @ W_true

        unc_ref = jnp.ones_like(ref) * 0.01
        unc_mov = jnp.ones_like(mov) * 0.01

        (_, G_rec, W_rec), (var, _) = nonrigid_align(
            ref, mov, 0.01, 1.0, kvar, 100, 1e-8,
            unc_ref=unc_ref, unc_mov=unc_mov,
        )

        # Verify transformed points match reference
        ref_recovered = mov + G_rec @ W_rec
        assert jnp.allclose(ref_recovered, ref, atol=0.15)


