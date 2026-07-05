import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from .util import mahalanobis_dist
from .util import sqdist


__all__ = [
    "MatchingMatrix",
    "expectation",
    "expectation_weighted",
    "expectation_masked",
]


type MatchingMatrix = Float[Array, "m n"]


def expectation(
    x: Float[Array, "n d"],
    y_t: Float[Array, "m d"],
    var: Float[Array, ""],
    w: float,
    moving_weights: Float[Array, " m"] | None = None,
    mask: Float[Array, "m n"] | None = None,
    unc_x: Float[Array, "n d"] | None = None,
    unc_y: Float[Array, "m d"] | None = None,
) -> MatchingMatrix:
    """Do a single expectation step of the CPD algorithm.

    This computes the matching matrix (posterior responsibilities) between
    target points ``x`` and transformed moving points ``y_t``. Supports
    optional mixing weights, pairwise masking, and per-point diagonal
    uncertainties.

    Args:
        x (Float[Array, "n d"]): target point set
        y_t (Float[Array, "m d"]): source (moving) point set, transformed
        var (Float[Array, ""]): variance of the Gaussian kernel
        w (float): outlier probability
        moving_weights (Float[Array, " m"] | None): optional per-point weights
            for the source points (arbitrary positive values). If None, uniform
            weights are used.
        mask (Float[Array, "m n"] | None): optional mask matrix where nonzero
            entries indicate valid matches.
        unc_x (Float[Array, "n d"] | None): optional per-point, per-dimension
            variances for target points.
        unc_y (Float[Array, "m d"] | None): optional per-point, per-dimension
            variances for moving points.

    Returns:
        MatchingMatrix: (m x n) matrix of matching probabilities.
    """
    n, d = x.shape
    m, _ = y_t.shape

    if unc_x is None and unc_y is None:
        # Fast path: original formula, no uncertainty overhead.
        d_t = sqdist(x, y_t).transpose()  # (m, n)
        top = jnp.exp(jnp.negative(jnp.divide(d_t, 2 * var)))
        if moving_weights is not None:
            top = moving_weights[:, None] * top
            weight_factor = jnp.sum(moving_weights)
        else:
            weight_factor = m
        outl_term = jnp.divide(w, 1.0 - w) * jnp.divide(
            jnp.float_power(2 * jnp.pi * var, d / 2) * weight_factor, n
        )
        if mask is not None:
            top = jnp.where(mask > 0, top, 0)
            n_mpr = jnp.sum(mask, axis=0, keepdims=True)
            n_msk = jnp.sum(mask)
            outl_term = outl_term * (n_mpr / n_msk)
        bot = jnp.add(
            jnp.clip(jnp.sum(top, axis=0, keepdims=True), jnp.finfo(x.dtype).eps),
            outl_term,
        )
        return jnp.divide(top, bot)
    else:
        # General path: Mahalanobis distance with per-point uncertainties.
        assert unc_x is not None and unc_y is not None
        mahal, log_det = mahalanobis_dist(x, unc_x, y_t, unc_y, var)
        # mahal, log_det are (n, m); transpose to (m, n) convention.
        mahal = mahal.T
        log_det = log_det.T  # (m, n), log(|Cov_{mn}|) per pair

        # Numerator in log domain: log(α_{mn})
        # Paper formula: α_{mn} = π_m · |2π·Cov_{mn}|^{-1/2} · exp(-0.5·mahal)
        # log(α) = log(π_m) - 0.5·d·log(2π) - 0.5·log_det - 0.5·mahal
        log_top = -0.5 * mahal - 0.5 * log_det - 0.5 * d * jnp.log(2 * jnp.pi)
        if moving_weights is not None:
            log_top = log_top + jnp.log(moving_weights[:, None])

        # Mask: set forbidden entries to -inf in log domain.
        if mask is not None:
            log_top = jnp.where(mask > 0, log_top, -jnp.inf)

        # Outlier term: outl_n = (w/(1-w)) · |M_n| / N_mask
        # (no covariance normalization — the normalization is in α).
        if moving_weights is not None:
            weight_factor = jnp.sum(moving_weights)
        else:
            weight_factor = m

        if mask is not None:
            n_mpr = jnp.sum(mask, axis=0, keepdims=True)  # (1, n)
            n_msk = jnp.sum(mask)
            outl_term = jnp.divide(w, 1.0 - w) * weight_factor / n * (
                n_mpr / n_msk
            )  # (1, n)
        else:
            outl_term = jnp.divide(w, 1.0 - w) * weight_factor / n

        # Convert from log domain to probability.
        top = jnp.exp(log_top)
        bot = jnp.add(
            jnp.clip(jnp.sum(top, axis=0, keepdims=True), jnp.finfo(x.dtype).eps),
            outl_term,
        )
        return jnp.divide(top, bot)


def expectation_weighted(
    x: Float[Array, "n d"],
    y_t: Float[Array, "m d"],
    var: Float[Array, ""],
    w: float,
    alpha_m: Float[Array, " m"],
) -> MatchingMatrix:
    """Do a single expectation step of the CPD algorithm, with per-point
    weightings for the source point set.

    Args:
        x (Float[Array, "n d"]): target point set
        y_t (Float[Array, "m d"]): source (moving) point set
        var (Float[Array, ""]): variance of the Gaussian kernel
        w (float): outlier probability
        alpha_m (Float[Array, " m"]): per-point weightings for the source
            points (arbitrary positive values)

    Returns:
        MatchingMatrix: (m x n) matrix of matching probabilities.
    """
    return expectation(x, y_t, var, w, moving_weights=alpha_m)


def expectation_masked(
    x: Float[Array, "n d"],
    y_t: Float[Array, "m d"],
    var: Float[Array, ""],
    w: float,
    mask: Float[Array, "m n"],
) -> MatchingMatrix:
    """Do a single expectation step of the CPD algorithm where only some of
    the entries in the matching matrix are allowed.

    Args:
        x (Float[Array, "n d"]): target point set
        y_t (Float[Array, "m d"]): source (moving) point set
        var (Float[Array, ""]): variance of the Gaussian kernel
        w (float): outlier probability
        mask (Float[Array, "m n"]): mask where nonzero entries indicate valid
            matches

    Returns:
        MatchingMatrix: (m x n) matrix of matching probabilities.
    """
    return expectation(x, y_t, var, w, mask=mask)
