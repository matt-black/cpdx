"""Rigid + Non-Rigid (RNR) registration.

Jointly estimates a global similarity transform (rotation, scale, translation)
and a smooth non-rigid displacement field within a single EM framework.

References
---
Based on Section 10 of the unified CPD framework paper
("Combined Rigid and Non-Rigid Registration").
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

from ._matching import MatchingMatrix
from ._matching import expectation
from ._matching import expectation_masked
from ._matching import expectation_weighted
from .nonrigid import (
    interpolate_variance as _nonrigid_interpolate_variance,
    update_transform as _nonrigid_update_transform,
    update_transform_uncertainty as _nonrigid_update_transform_uncertainty,
)
from .rigid import (
    update_transform_uncertainty as _rigid_update_transform_uncertainty,
    update_variance_uncertainty as _rigid_update_variance_uncertainty,
)
from .util import sqdist


__all__ = [
    "RotationMatrix",
    "ScalingTerm",
    "Translation",
    "KernelMatrix",
    "CoeffMatrix",
    "TransformParams",
    "align",
    "align_fixed_iter",
    "transform",
    "interpolate",
    "interpolate_variance",
    "maximization",
    "maximization_uncertainty",
]


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

type RotationMatrix = Float[Array, "d d"]
type ScalingTerm = Float[Array, ""]
type Translation = Float[Array, " d"]
type KernelMatrix = Float[Array, "m m"]
type CoeffMatrix = Float[Array, "m d"]
type TransformParams = tuple[
    MatchingMatrix,
    RotationMatrix,
    ScalingTerm,
    Translation,
    KernelMatrix,
    CoeffMatrix,
]


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def transform(
    y: Float[Array, "m d"],
    R: RotationMatrix,
    s: ScalingTerm,
    t: Translation,
    G: KernelMatrix,
    W: CoeffMatrix,
) -> Float[Array, "m d"]:
    """Apply the combined rigid + non-rigid transformation.

    Args:
        y: `d`-dimensional points to be transformed (m, d).
        R: `d`-dimensional rotation matrix (d, d).
        s: scalar isotropic scaling term.
        t: translation vector (d,).
        G: kernel Gram matrix (m, m).
        W: GP coefficient matrix (m, d).

    Returns:
        Transformed points ``s * (y @ R.T) + t + G @ W``.
    """
    return s * (y @ R.T) + t[None, :] + G @ W


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------


def _compute_weights(
    P: MatchingMatrix,
    var: Float[Array, ""],
    unc_x: Float[Array, "n d"],
    unc_y: Float[Array, "m d"],
) -> Float[Array, "m n"]:
    """Compute scalar effective weights w_{mn} from responsibilities.

    Uses the average-precision approximation:
        w_bar[m,n] = (1/d) * sum_k 1 / (var + unc_x[n,k] + unc_y[m,k])
    and sets w_{mn} = P_{mn} * w_bar[m,n].

    Args:
        P: Matching matrix (m, n).
        var: Current isotropic variance.
        unc_x: Target per-point, per-dimension variances (n, d).
        unc_y: Moving per-point, per-dimension variances (m, d).

    Returns:
        Effective weight matrix (m, n).
    """
    cov_diag = var + unc_x[None, :, :] + unc_y[:, None, :]  # (m, n, d)
    w_bar = jnp.sum(1.0 / cov_diag, axis=2) / cov_diag.shape[2]  # (m, n)
    return P * w_bar


# ---------------------------------------------------------------------------
# Rigid substep
# ---------------------------------------------------------------------------


def _update_rigid(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    P: MatchingMatrix,
    V: CoeffMatrix,
) -> tuple[RotationMatrix, ScalingTerm, Translation]:
    """Rigid substep without uncertainty (baseline).

    Minimises Σ_{m,n} P_{mn} || (ref[n] - V[m]) - s R mov[m] - t ||²
    using weighted Procrustes on compensated targets r_{mn} = ref[n] - V[m].

    Args:
        ref: Target points (n, d).
        mov: Moving (source) points (m, d).
        P: Matching matrix (m, n).
        V: Current displacement field G @ W (m, d).

    Returns:
        (R, s, t)
    """
    _, d = ref.shape

    # Centroids
    W_tot = jnp.sum(P)
    P1 = jnp.sum(P, axis=1)  # (m,)
    Pt1 = jnp.sum(P, axis=0)  # (n,)

    x_bar = jnp.einsum("m,mj->j", P1, mov) / W_tot  # (d,)
    ref_bar = jnp.einsum("n,nj->j", Pt1, ref) / W_tot  # (d,)
    V_bar = jnp.einsum("m,mj->j", P1, V) / W_tot  # (d,)
    r_bar = ref_bar - V_bar  # (d,)

    # Centered
    mov_c = mov - x_bar[None, :]  # (m, d)

    # Cross-covariance
    # Matches rigid.py convention: H_{jk} = Σ_{m,n} P_{mn} * r_mn_c[m,n,j] * mov_c[m,k]
    r_mn_c = ref[None, :, :] - V[:, None, :] - r_bar[None, None, :]  # (m, n, d)
    H = jnp.einsum("mn,mk,mnj->jk", P, mov_c, r_mn_c)  # (d, d)

    # SVD
    U, _, Vt = jnp.linalg.svd(H)
    C = jnp.diag(
        jnp.concatenate(
            [jnp.ones((d - 1,)), jnp.linalg.det(U @ Vt)[None]], axis=0
        )
    )
    R = U @ C @ Vt

    # Scale: s = tr(H.T @ R) / Σ_{m,n} P_{mn} ||mov_c[m]||²
    # Matches rigid.py convention: s = trace(A.T @ R) / trace(y_hat.T @ diag(P1) @ y_hat)
    s = jnp.trace(H.T @ R) / jnp.sum(P1[:, None] * jnp.square(mov_c))

    # Translation
    t = r_bar - s * (R @ x_bar)

    return R, s, t


def _update_rigid_uncertainty(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    P: MatchingMatrix,
    V: CoeffMatrix,
    var: Float[Array, ""],
    unc_x: Float[Array, "n d"],
    unc_y: Float[Array, "m d"],
) -> tuple[RotationMatrix, ScalingTerm, Translation]:
    """Rigid substep with per-point diagonal uncertainties.

    Minimises Σ_{m,n} P_{mn} * w_bar_{mn} *
        || (ref[n] - V[m]) - s R mov[m] - t ||²
    where w_bar uses the average-precision approximation.

    Args:
        ref: Target points (n, d).
        mov: Moving (source) points (m, d).
        P: Matching matrix (m, n).
        V: Current displacement field G @ W (m, d).
        var: Current isotropic variance.
        unc_x: Target per-point variances (n, d).
        unc_y: Moving per-point variances (m, d).

    Returns:
        (R, s, t)
    """
    _, d = ref.shape

    w_mn = _compute_weights(P, var, unc_x, unc_y)  # (m, n)

    # Centroids
    W_tot = jnp.sum(w_mn)
    P1 = jnp.sum(w_mn, axis=1)  # (m,)
    Pt1 = jnp.sum(w_mn, axis=0)  # (n,)

    x_bar = jnp.einsum("m,mj->j", P1, mov) / W_tot  # (d,)
    ref_bar = jnp.einsum("n,nj->j", Pt1, ref) / W_tot  # (d,)
    V_bar = jnp.einsum("m,mj->j", P1, V) / W_tot  # (d,)
    r_bar = ref_bar - V_bar  # (d,)

    # Centered
    mov_c = mov - x_bar[None, :]  # (m, d)

    # Cross-covariance
    # Matches rigid.py convention: H_{jk} = Σ_{m,n} w_{mn} * r_mn_c[m,n,j] * mov_c[m,k]
    r_mn_c = ref[None, :, :] - V[:, None, :] - r_bar[None, None, :]  # (m, n, d)
    H = jnp.einsum("mn,mk,mnj->jk", w_mn, mov_c, r_mn_c)  # (d, d)

    # SVD
    U, _, Vt = jnp.linalg.svd(H)
    C = jnp.diag(
        jnp.concatenate(
            [jnp.ones((d - 1,)), jnp.linalg.det(U @ Vt)[None]], axis=0
        )
    )
    R = U @ C @ Vt

    # Scale: s = tr(H.T @ R) / Σ_{m,n} w_{mn} ||mov_c[m]||²
    # Matches rigid.py convention: s = trace(A.T @ R) / trace(y_hat.T @ diag(P1) @ y_hat)
    s = jnp.trace(H.T @ R) / jnp.sum(P1[:, None] * jnp.square(mov_c))

    # Translation
    t = r_bar - s * (R @ x_bar)

    return R, s, t


# ---------------------------------------------------------------------------
# M-step (maximization)
# ---------------------------------------------------------------------------


def maximization(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    P: MatchingMatrix,
    G: KernelMatrix,
    var: Float[Array, ""],
    regularization_param: float,
    tolerance: float,
    inner_iter: int = 2,
    burn_in: Bool[Array, ""] | bool = False,
) -> tuple[tuple[RotationMatrix, ScalingTerm, Translation, CoeffMatrix], Float[Array, ""]]:
    """M-step without per-point uncertainties.

    Args:
        ref: Target point set (n, d).
        mov: Source point set (m, d).
        P: Matching matrix (m, n).
        G: Kernel Gram matrix (m, m).
        var: Current isotropic variance.
        regularization_param: GP regularization parameter λ.
        tolerance: Termination tolerance.
        inner_iter: Number of rigid↔non-rigid alternations.
        burn_in: JAX boolean — if True, skip non-rigid substep.

    Returns:
        ((R, s, t, W), variance)
    """
    m, d = mov.shape

    def inner_step(
        state: tuple[
            tuple[RotationMatrix, ScalingTerm, Translation],
            CoeffMatrix,
        ],
        _: int,
    ) -> tuple[
        tuple[
            tuple[RotationMatrix, ScalingTerm, Translation],
            CoeffMatrix,
        ],
        None,
    ]:
        (R, s, t), W = state

        # Displacement field
        V = G @ W  # (m, d)

        # Rigid substep
        R, s, t = _update_rigid(ref, mov, P, V)

        # Non-rigid substep — use jax.lax.cond
        def _do_nonrigid(_):
            mov_rigid = s * (mov @ R.T) + t[None, :]
            W_new = _nonrigid_update_transform(
                ref, mov_rigid, P, G, var, regularization_param
            )
            return W_new

        def _skip_nonrigid(_):
            return W

        W = jax.lax.cond(
            burn_in, _skip_nonrigid, _do_nonrigid, None
        )

        return ((R, s, t), W), None

    W_init = jnp.zeros_like(mov)
    ((R, s, t), W), _ = jax.lax.scan(
        inner_step,
        ((jnp.eye(d), jnp.array(1.0), jnp.zeros((d,))), W_init),
        length=inner_iter,
    )

    # Variance update
    y_t = transform(mov, R, s, t, G, W)
    residuals_sq = sqdist(ref, y_t).T  # (m, n)
    N = jnp.sum(P)
    new_var = jnp.sum(P * residuals_sq) / (N * d)
    new_var = jax.lax.select(new_var > 0, new_var, tolerance / 10.0)

    return (R, s, t, W), new_var


def maximization_uncertainty(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    P: MatchingMatrix,
    G: KernelMatrix,
    var: Float[Array, ""],
    regularization_param: float,
    tolerance: float,
    unc_x: Float[Array, "n d"],
    unc_y: Float[Array, "m d"],
    inner_iter: int = 2,
    burn_in: Bool[Array, ""] | bool = False,
) -> tuple[tuple[RotationMatrix, ScalingTerm, Translation, CoeffMatrix], Float[Array, ""]]:
    """M-step with per-point diagonal uncertainties.

    Args:
        ref: Target point set (n, d).
        mov: Source point set (m, d).
        P: Matching matrix (m, n).
        G: Kernel Gram matrix (m, m).
        var: Current isotropic variance.
        regularization_param: GP regularization parameter λ.
        tolerance: Termination tolerance.
        unc_x: Target per-point variances (n, d).
        unc_y: Moving per-point variances (m, d).
        inner_iter: Number of rigid↔non-rigid alternations.
        burn_in: JAX boolean — if True, skip non-rigid substep.

    Returns:
        ((R, s, t, W), variance)
    """
    m, d = mov.shape

    def inner_step(
        state: tuple[
            tuple[RotationMatrix, ScalingTerm, Translation],
            CoeffMatrix,
        ],
        _: int,
    ) -> tuple[
        tuple[
            tuple[RotationMatrix, ScalingTerm, Translation],
            CoeffMatrix,
        ],
        None,
    ]:
        (R, s, t), W = state

        # Displacement field
        V = G @ W  # (m, d)

        # Rigid substep with uncertainty
        R, s, t = _update_rigid_uncertainty(ref, mov, P, V, var, unc_x, unc_y)

        # Non-rigid substep — use jax.lax.cond
        def _do_nonrigid(_):
            mov_rigid = s * (mov @ R.T) + t[None, :]
            W_new = _nonrigid_update_transform_uncertainty(
                ref, mov_rigid, P, G, var, regularization_param, unc_x, unc_y
            )
            return W_new

        def _skip_nonrigid(_):
            return W

        W = jax.lax.cond(
            burn_in, _skip_nonrigid, _do_nonrigid, None
        )

        return ((R, s, t), W), None

    W_init = jnp.zeros_like(mov)
    ((R, s, t), W), _ = jax.lax.scan(
        inner_step,
        ((jnp.eye(d), jnp.array(1.0), jnp.zeros((d,))), W_init),
        length=inner_iter,
    )

    # Variance update with uncertainty
    y_t = transform(mov, R, s, t, G, W)
    new_var = _rigid_update_variance_uncertainty(ref, y_t, P, tolerance, unc_x, unc_y)

    return (R, s, t, W), new_var


# ---------------------------------------------------------------------------
# Alignment (EM loop)
# ---------------------------------------------------------------------------


def align(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    outlier_prob: float,
    regularization_param: float,
    kernel_var: float,
    max_iter: int,
    tolerance: float,
    unc_ref: Float[Array, "n d"] | None = None,
    unc_mov: Float[Array, "m d"] | None = None,
    moving_weights: Float[Array, " m"] | None = None,
    mask: Float[Array, "m n"] | None = None,
    inner_iter: int = 2,
    burn_in: int = 0,
) -> tuple[TransformParams, tuple[Float[Array, ""], int]]:
    """Align moving points to reference using combined rigid + non-rigid transform.

    Jointly estimates a global similarity transform (rotation, scale, translation)
    and a smooth non-rigid displacement field using an EM algorithm with an
    inner alternating loop between rigid and non-rigid updates.

    Args:
        ref: Reference (target) points (n, d).
        mov: Moving (source) points (m, d).
        outlier_prob: Outlier probability in [0, 1].
        regularization_param: GP regularization parameter λ.
        kernel_var: Gaussian kernel width β².
        max_iter: Maximum number of EM iterations.
        tolerance: Convergence tolerance on variance.
        unc_ref: Optional per-point variances for reference points (n, d).
        unc_mov: Optional per-point variances for moving points (m, d).
        moving_weights: Optional per-point mixing weights for moving points.
        mask: Optional pairwise validity mask (m, n).
        inner_iter: Inner rigid↔non-rigid alternations per M-step (default: 2).
        burn_in: Number of initial EM iterations with W=0 (pure rigid phase).

    Returns:
        ``(P, R, s, t, G, W), (final_variance, iterations_run)``.
    """
    n, d = ref.shape
    m, _ = mov.shape
    var_i = jnp.sum(sqdist(ref, mov)) / (m * n * d)
    has_unc = unc_ref is not None and unc_mov is not None

    # Compute kernel matrix
    G = jnp.exp(-sqdist(mov, mov) / (2 * kernel_var))

    def cond_fun(
        a: tuple[
            tuple[RotationMatrix, ScalingTerm, Translation, CoeffMatrix, MatchingMatrix],
            tuple[Float[Array, ""], int],
        ],
    ) -> Bool:
        _, (var, iter_num) = a
        return jnp.logical_and(var > tolerance, iter_num < max_iter)

    is_burn_in_init = jnp.array(0) < jnp.array(burn_in)

    # Build a single body_fun to handle both uncertainty paths.
    # When has_unc is False and (moving_weights, mask) are None, we use
    # a simpler E-step path.  This is dispatched at trace time via Python if,
    # since has_unc / moving_weights / mask are concrete Python values.

    if has_unc:
        assert unc_ref is not None and unc_mov is not None

        def body_fun(
            a: tuple[
                tuple[RotationMatrix, ScalingTerm, Translation, CoeffMatrix, MatchingMatrix],
                tuple[Float[Array, ""], int],
            ],
        ) -> tuple[
            tuple[RotationMatrix, ScalingTerm, Translation, CoeffMatrix, MatchingMatrix],
            tuple[Float[Array, ""], int],
        ]:
            (R, s, t, W, _), (var, iter_num) = a
            is_burn_in = iter_num < burn_in

            # During burn-in, use rigid-only transform for E-step.
            mov_t_full = transform(mov, R, s, t, G, W)
            mov_t_rigid = s * (mov @ R.T) + t[None, :]
            mov_t = jax.lax.select(is_burn_in, mov_t_rigid, mov_t_full)

            P = expectation(
                ref, mov_t, var, outlier_prob,
                moving_weights=moving_weights, mask=mask,
                unc_x=unc_ref, unc_y=unc_mov,
            )
            (R, s, t, W), new_var = maximization_uncertainty(
                ref, mov, P, G, var, regularization_param, tolerance,
                unc_ref, unc_mov,
                inner_iter=inner_iter, burn_in=is_burn_in,
            )
            return (R, s, t, W, P), (new_var, iter_num + 1)
    else:
        if moving_weights is None:
            if mask is None:
                use_mask = False
                use_weights = False
            else:
                use_mask = True
                use_weights = False
        else:
            use_mask = False
            use_weights = True

        def body_fun(
            a: tuple[
                tuple[RotationMatrix, ScalingTerm, Translation, CoeffMatrix, MatchingMatrix],
                tuple[Float[Array, ""], int],
            ],
        ) -> tuple[
            tuple[RotationMatrix, ScalingTerm, Translation, CoeffMatrix, MatchingMatrix],
            tuple[Float[Array, ""], int],
        ]:
            (R, s, t, W, _), (var, iter_num) = a
            is_burn_in = iter_num < burn_in

            mov_t = transform(mov, R, s, t, G, W)

            if use_weights:
                assert moving_weights is not None
                P = expectation_weighted(ref, mov_t, var, outlier_prob, moving_weights)
            elif use_mask:
                assert mask is not None
                P = expectation_masked(ref, mov_t, var, outlier_prob, mask)
            else:
                P = expectation(ref, mov_t, var, outlier_prob)

            (R, s, t, W), new_var = maximization(
                ref, mov, P, G, var, regularization_param, tolerance,
                inner_iter=inner_iter, burn_in=is_burn_in,
            )
            return (R, s, t, W, P), (new_var, iter_num + 1)

    (R, s, t, W, P), (var_f, num_iter) = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (
            (
                jnp.eye(d),
                jnp.array(1.0),
                jnp.zeros((d,)),
                jnp.zeros((m, d)),
                jnp.zeros((m, n)),
            ),
            (var_i, 0),
        ),
    )

    return (P, R, s, t, G, W), (var_f, num_iter)


def align_fixed_iter(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    outlier_prob: float,
    regularization_param: float,
    kernel_var: float,
    num_iter: int,
    unc_ref: Float[Array, "n d"] | None = None,
    unc_mov: Float[Array, "m d"] | None = None,
    moving_weights: Float[Array, " m"] | None = None,
    mask: Float[Array, "m n"] | None = None,
    inner_iter: int = 2,
    burn_in: int = 0,
) -> tuple[TransformParams, Float[Array, " {num_iter}"]]:
    """Align moving points to reference using combined rigid + non-rigid transform.

    Fixed-iteration version that returns variance at each step.

    Args:
        ref: Reference (target) points (n, d).
        mov: Moving (source) points (m, d).
        outlier_prob: Outlier probability in [0, 1].
        regularization_param: GP regularization parameter λ.
        kernel_var: Gaussian kernel width β².
        num_iter: Number of EM iterations to run.
        unc_ref: Optional per-point variances for reference points (n, d).
        unc_mov: Optional per-point variances for moving points (m, d).
        moving_weights: Optional per-point mixing weights for moving points.
        mask: Optional pairwise validity mask (m, n).
        inner_iter: Inner rigid↔non-rigid alternations per M-step (default: 2).
        burn_in: Number of initial EM iterations with W=0 (pure rigid phase).

    Returns:
        ``(P, R, s, t, G, W), variances`` where variances has shape ``(num_iter,)``.
    """
    n, d = ref.shape
    m, _ = mov.shape
    var_i = (jnp.sum(sqdist(ref, mov)) / (m * n * d)).item()
    has_unc = unc_ref is not None and unc_mov is not None

    # Compute kernel matrix
    G = jnp.exp(-sqdist(mov, mov) / (2 * kernel_var))

    # Use a counter in state to track iteration for burn_in
    def scan_fun(
        state: tuple[
            tuple[RotationMatrix, ScalingTerm, Translation, CoeffMatrix, MatchingMatrix],
            Float[Array, ""],  # var
            int,  # iter_count
        ],
        _: int,
    ):
        (R, s, t, W, P), var, cnt = state

        is_burn_in = cnt < burn_in
        mov_t_full = transform(mov, R, s, t, G, W)
        mov_t_rigid = s * (mov @ R.T) + t[None, :]
        mov_t = jax.lax.select(is_burn_in, mov_t_rigid, mov_t_full)

        if has_unc:
            assert unc_ref is not None and unc_mov is not None
            P = expectation(
                ref, mov_t, var, outlier_prob,
                moving_weights=moving_weights, mask=mask,
                unc_x=unc_ref, unc_y=unc_mov,
            )
            (R, s, t, W), new_var = maximization_uncertainty(
                ref, mov, P, G, var, regularization_param, 1e-6,
                unc_ref, unc_mov,
                inner_iter=inner_iter, burn_in=is_burn_in,
            )
        else:
            if moving_weights is None:
                if mask is None:
                    P = expectation(ref, mov_t, var, outlier_prob)
                else:
                    P = expectation_masked(ref, mov_t, var, outlier_prob, mask)
            else:
                P = expectation_weighted(
                    ref, mov_t, var, outlier_prob, moving_weights
                )
            (R, s, t, W), new_var = maximization(
                ref, mov, P, G, var, regularization_param, 1e-6,
                inner_iter=inner_iter, burn_in=is_burn_in,
            )

        return ((R, s, t, W, P), new_var, cnt + 1), new_var

    ((R, s, t, W, P), _, _), varz = jax.lax.scan(
        scan_fun,
        (
            (
                jnp.eye(d),
                jnp.array(1.0),
                jnp.zeros((d,)),
                jnp.zeros((m, d)),
                jnp.zeros((m, n)),
            ),
            var_i,
            0,
        ),
        length=num_iter,
    )

    return (P, R, s, t, G, W), varz


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def interpolate(
    mov: Float[Array, "m d"],
    interp: Float[Array, "n d"],
    R: RotationMatrix,
    s: ScalingTerm,
    t: Translation,
    W: CoeffMatrix,
    kernel_var: float,
) -> Float[Array, "n d"]:
    """Interpolate the combined transformation to new query points.

    Computes ``T(z) = s * (z @ R.T) + t + G(z, mov) @ W``.

    Args:
        mov: Original moving (control) points (m, d).
        interp: Query points to interpolate at (n, d).
        R: Fitted rotation matrix (d, d).
        s: Fitted scale.
        t: Fitted translation (d,).
        W: Fitted GP coefficient matrix (m, d).
        kernel_var: Gaussian kernel width β² (must match fitting).

    Returns:
        Transformed query points (n, d).
    """
    # Cross-kernel between query and control points
    G_im = jnp.exp(-sqdist(interp, mov) / (2 * kernel_var))  # (n, m)
    return s * (interp @ R.T) + t[None, :] + G_im @ W


def interpolate_variance(
    mov: Float[Array, "m d"],
    interp: Float[Array, "n d"],
    P: MatchingMatrix,
    G: KernelMatrix,
    kernel_var: float,
    regularization_param: float,
    var: float,
    method: str = "cholesky",
    rank: int | None = None,
    eps: float = 1e-3,
) -> Float[Array, " n"]:
    """Compute posterior variance of the interpolated displacement field.

    Delegates to :func:`nonrigid.interpolate_variance` for the GP component.
    The returned variance reflects uncertainty in the non-rigid displacement
    only; it does not include rigid-parameter uncertainty or point-wise noise.

    Args:
        mov: Moving (control) points from alignment (m, d).
        interp: Query points for interpolation (n, d).
        P: Posterior probability matrix from alignment (m, n_ref).
        G: Gram matrix between moving points (m, m).
        kernel_var: Gaussian kernel width β².
        regularization_param: Regularization parameter λ.
        var: Converged variance σ² from alignment.
        method: Numerical method — "cholesky" or "low_rank".
        rank: Rank K for low-rank approximation.
        eps: Small nugget for numerical stability.

    Returns:
        Per-point, per-coordinate posterior variance of shape (n,).
    """
    return _nonrigid_interpolate_variance(
        mov, interp, P, G, kernel_var,
        regularization_param, var, method, rank, eps,
    )
