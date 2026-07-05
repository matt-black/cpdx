import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

from ._matching import MatchingMatrix
from ._matching import expectation
from ._matching import expectation_masked
from ._matching import expectation_weighted
from .util import sqdist


__all__ = [
    "RotationMatrix",
    "ScalingTerm",
    "Translation",
    "TransformParams",
    "align",
    "align_fixed_iter",
    "transform",
    "maximization",
    "maximization_uncertainty",
]


type RotationMatrix = Float[Array, "d d"]
type ScalingTerm = Float[Array, ""]
type Translation = Float[Array, " d"]
type TransformParams = tuple[
    MatchingMatrix, RotationMatrix, ScalingTerm, Translation
]


def align(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    outlier_prob: float,
    max_iter: int,
    tolerance: float,
    unc_ref: Float[Array, "n d"] | None = None,
    unc_mov: Float[Array, "m d"] | None = None,
    moving_weights: Float[Array, " m"] | None = None,
    mask: Float[Array, "m n"] | None = None,
) -> tuple[
    tuple[MatchingMatrix, RotationMatrix, ScalingTerm, Translation],
    tuple[Float[Array, ""], int],
]:
    """Align the moving points onto the reference points by rigid transform.

    Args:
        ref (Float[Array, "n d"]): reference points
        mov (Float[Array, "m d"]): moving points
        outlier_prob (float): outlier probability, should be in range [0,1].
        max_iter (int): maximum # of iterations to optimize for.
        tolerance (float): tolerance for matching variance, below which the algorithm will terminate.
        unc_ref (Float[Array, "n d"] | None): optional per-point, per-dimension
            variances for reference (target) points.
        unc_mov (Float[Array, "m d"] | None): optional per-point, per-dimension
            variances for moving points.
        moving_weights (Float[Array, " m"] | None): optional per-point weights for source points (arbitrary positive values). If None, uniform weights are used.
        mask (Float[Array, "m n"] | None): optional mask matrix where nonzero entries indicate valid matches.

    Returns:
        tuple[TransformParams, tuple[Float[Array, ""], int]]: the fitted transform parameters (a rotation matrix, scaling term, and translation) along with the final variance and the number of iterations that the algorithm was run for.
    """
    n, d = ref.shape
    m, _ = mov.shape
    var_i = jnp.sum(sqdist(ref, mov)) / (m * n * d)
    has_unc = unc_ref is not None and unc_mov is not None

    def cond_fun(
        a: tuple[
            tuple[RotationMatrix, ScalingTerm, Translation, MatchingMatrix],
            tuple[Float[Array, ""], int],
        ],
    ) -> Bool:
        _, (var, iter_num) = a
        return jnp.logical_and(var > tolerance, iter_num < max_iter)

    def body_fun(
        a: tuple[
            tuple[RotationMatrix, ScalingTerm, Translation, MatchingMatrix],
            tuple[Float[Array, ""], int],
        ],
    ) -> tuple[
        tuple[RotationMatrix, ScalingTerm, Translation, MatchingMatrix],
        tuple[Float[Array, ""], int],
    ]:
        (R, s, t, P), (var, iter_num) = a
        mov_t = transform(mov, R, s, t)
        if has_unc:
            assert unc_ref is not None and unc_mov is not None
            P = expectation(
                ref, mov_t, var, outlier_prob,
                moving_weights=moving_weights, mask=mask,
                unc_x=unc_ref, unc_y=unc_mov,
            )
            (R, s, t), new_var = maximization_uncertainty(
                ref, mov, P, tolerance, var, unc_ref, unc_mov
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
            (R, s, t), new_var = maximization(ref, mov, P, tolerance)
        return (R, s, t, P), (new_var, iter_num + 1)

    (R, s, t, P), (var_f, num_iter) = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (
            (
                jnp.eye(d),
                jnp.array(1.0),
                jnp.zeros((d,)),
                jnp.zeros((m, n)),
            ),
            (var_i, 0),
        ),
    )
    return (P, R, s, t), (var_f, num_iter)


def align_fixed_iter(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    outlier_prob: float,
    num_iter: int,
    unc_ref: Float[Array, "n d"] | None = None,
    unc_mov: Float[Array, "m d"] | None = None,
    moving_weights: Float[Array, " m"] | None = None,
    mask: Float[Array, "m n"] | None = None,
) -> tuple[
    tuple[MatchingMatrix, RotationMatrix, ScalingTerm, Translation],
    Float[Array, " {num_iter}"],
]:
    """Align the moving points onto the reference points by rigid transform.

    Args:
        ref (Float[Array, "n d"]): reference points
        mov (Float[Array, "m d"]): moving points
        outlier_prob (float): outlier probability, should be in range [0,1].
        num_iter (int): # of iterations to optimize for.
        unc_ref (Float[Array, "n d"] | None): optional per-point, per-dimension
            variances for reference (target) points.
        unc_mov (Float[Array, "m d"] | None): optional per-point, per-dimension
            variances for moving points.
        moving_weights (Float[Array, " m"] | None): optional per-point weights for source points (arbitrary positive values). If None, uniform weights are used.
        mask (Float[Array, "m n"] | None): optional mask matrix where nonzero entries indicate valid matches.

    Returns:
        tuple[TransformParams, Float[Array, " {num_iter}"]]: the fitted transform parameters (a rotation matrix, scaling term, and translation) along with the variance at each step of the optimization.
    """
    # initialize variance
    n, d = ref.shape
    m, _ = mov.shape
    var_i = (jnp.sum(sqdist(ref, mov)) / (m * n * d)).item()
    has_unc = unc_ref is not None and unc_mov is not None

    def scan_fun(
        a: tuple[
            tuple[RotationMatrix, ScalingTerm, Translation, MatchingMatrix],
            Float[Array, ""],
        ],
        _,
    ) -> tuple[
        tuple[
            tuple[RotationMatrix, ScalingTerm, Translation, MatchingMatrix],
            Float[Array, ""],
        ],
        Float[Array, ""],
    ]:
        (R, s, t, P), var = a
        mov_t = transform(mov, R, s, t)
        if has_unc:
            assert unc_ref is not None and unc_mov is not None
            p = expectation(
                ref, mov_t, var, outlier_prob,
                moving_weights=moving_weights, mask=mask,
                unc_x=unc_ref, unc_y=unc_mov,
            )
            (R, s, t), new_var = maximization_uncertainty(
                ref, mov, p, 1e-6, var, unc_ref, unc_mov
            )
        else:
            if moving_weights is None:
                if mask is None:
                    p = expectation(ref, mov_t, var, outlier_prob)
                else:
                    p = expectation_masked(ref, mov_t, var, outlier_prob, mask)
            else:
                p = expectation_weighted(
                    ref, mov_t, var, outlier_prob, moving_weights
                )
            (R, s, t), new_var = maximization(ref, mov, p, 1e-6)
        return ((R, s, t, P), new_var), new_var

    ((R, s, t, P), _), varz = jax.lax.scan(
        scan_fun,
        (
            (
                jnp.eye(d),
                jnp.array(1.0),
                jnp.zeros((d,)),
                jnp.zeros((m, n)),
            ),
            var_i,
        ),
        None,
        length=num_iter,
    )
    return (P, R, s, t), varz


def transform(
    y: Float[Array, "m d"],
    R: RotationMatrix,
    s: ScalingTerm,
    t: Translation,
) -> Float[Array, "m d"]:
    """Transform the input points by rigid transformation.

    Args:
        y (Float[Array, "m d"]): `d`-dimensional points to be transformed
        R (RotationMatrix): `d`-dimensional rotation matrix
        s (ScalingTerm): scalar, isotropic scaling term
        t (Translation): translation

    Returns:
        Float[Array, "m d"]: transformed points, `s * (y @ R.T) + t`
    """
    return s * (y @ R.T) + t


def maximization(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    P: Float[Array, "m n"],
    tolerance: float,
) -> tuple[tuple[RotationMatrix, ScalingTerm, Translation], Float[Array, ""]]:
    """Do a single M-step.

    Args:
        x (Float[Array, "n d"]): target point set
        y (Float[Array, "m d"]): source point set
        P (MatchingMatrix): matching matrix
        tolerance (float): termination tolerance

    Returns:
        tuple[tuple[RotationMatrix, ScalingTerm, Translation], Float[Array, ""]]: updated transform parameters, and variance.
    """
    _, d = x.shape
    N = jnp.sum(P)
    Pt1, P1 = jnp.sum(P, axis=0), jnp.sum(P, axis=1)
    mu_x = jnp.divide(x.T @ Pt1, N)
    mu_y = jnp.divide(y.T @ P1, N)
    x_hat, y_hat = x - mu_x, y - mu_y
    A = x_hat.T @ P.T @ y_hat
    U, _, Vt = jnp.linalg.svd(A)
    C = jnp.diag(
        jnp.concatenate(
            [jnp.ones((d - 1,)), jnp.linalg.det(U @ Vt)[None]], axis=0
        )
    )
    R = U @ C @ Vt
    s = jnp.divide(
        jnp.trace(A.T @ R), jnp.trace(y_hat.T @ jnp.diag(P1) @ y_hat)
    )
    t = mu_x - s * (R @ mu_y)
    var = jnp.divide(
        jnp.trace(x_hat.T @ jnp.diag(Pt1) @ x_hat) - s * jnp.trace(A.T @ R),
        N * d,
    )
    return (R, s, t), jax.lax.select(var > 0, var, tolerance / 10.0)


def update_transform_uncertainty(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    P: Float[Array, "m n"],
    var: Float[Array, ""],
    unc_x: Float[Array, "n d"],
    unc_y: Float[Array, "m d"],
) -> tuple[RotationMatrix, ScalingTerm, Translation]:
    """Weighted Procrustes with per-dimension uncertainty weights.

    Computes a scalar effective weight per pair:
        w_bar[m,n] = (1/d) * sum_k 1 / (var + unc_x[n,k] + unc_y[m,k])
    and uses P_eff = P * w_bar for the weighted Procrustes solution.

    Args:
        x: Target point set (n, d).
        y: Source point set (m, d).
        P: Matching matrix (m, n).
        var: Current isotropic variance.
        unc_x: Target per-point, per-dimension variances (n, d).
        unc_y: Moving per-point, per-dimension variances (m, d).

    Returns:
        tuple[RotationMatrix, ScalingTerm, Translation]: rotation, scale, translation.
    """
    _, d = x.shape

    # Per-dimension covariance: cov[m, n, k] = var + unc_x[n, k] + unc_y[m, k]
    cov_diag = var + unc_x[None, :, :] + unc_y[:, None, :]  # (m, n, d)
    # Scalar effective weight: w_bar[m, n] = (1/d) * sum_k 1/cov[m,n,k]
    w_bar = jnp.sum(1.0 / cov_diag, axis=2) / d  # (m, n)
    # Weighted matching matrix
    P_eff = P * w_bar  # (m, n)

    # Standard Procrustes on P_eff
    N = jnp.sum(P_eff)
    Pt1, P1 = jnp.sum(P_eff, axis=0), jnp.sum(P_eff, axis=1)
    mu_x = jnp.divide(x.T @ Pt1, N)
    mu_y = jnp.divide(y.T @ P1, N)
    x_hat, y_hat = x - mu_x, y - mu_y
    A = x_hat.T @ P_eff.T @ y_hat
    U, _, Vt = jnp.linalg.svd(A)
    C = jnp.diag(
        jnp.concatenate(
            [jnp.ones((d - 1,)), jnp.linalg.det(U @ Vt)[None]], axis=0
        )
    )
    R = U @ C @ Vt
    s = jnp.divide(
        jnp.trace(A.T @ R), jnp.trace(y_hat.T @ jnp.diag(P1) @ y_hat)
    )
    t = mu_x - s * (R @ mu_y)
    return R, s, t


def update_variance_uncertainty(
    x: Float[Array, "n d"],
    y_t: Float[Array, "m d"],
    P: MatchingMatrix,
    tolerance: float,
    unc_x: Float[Array, "n d"],
    unc_y: Float[Array, "m d"],
) -> Float[Array, ""]:
    """Compute variance from residuals, subtracting uncertainty traces.

    σ²_new = max(ε, (1/(d·N_P)) · Σ P_{mn} · (||y_n - T(x_m)||² - tr(Σ_n) - tr(Γ_m)))

    Args:
        x: Target point set (n, d).
        y_t: Transformed moving points (m, d).
        P: Matching matrix (m, n).
        tolerance: Termination tolerance (used as floor).
        unc_x: Target per-point, per-dimension variances (n, d).
        unc_y: Moving per-point, per-dimension variances (m, d).

    Returns:
        Float[Array, ""]: Updated variance.
    """
    _, d = x.shape
    N = jnp.sum(P)

    # Squared residuals: ||x[n] - y_t[m]||², shape (m, n)
    residuals_sq = sqdist(x, y_t).T  # (m, n)

    # Trace of uncertainty per pair: sum_k(unc_x[n,k]) + sum_k(unc_y[m,k])
    trace_sum = jnp.sum(unc_y, axis=1)[:, None] + jnp.sum(unc_x, axis=1)[None, :]  # (m, n)

    new_var = jnp.sum(P * (residuals_sq - trace_sum)) / (N * d)
    return jax.lax.select(new_var > 0, new_var, tolerance / 10.0)


def maximization_uncertainty(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    P: Float[Array, "m n"],
    tolerance: float,
    var: Float[Array, ""],
    unc_x: Float[Array, "n d"],
    unc_y: Float[Array, "m d"],
) -> tuple[tuple[RotationMatrix, ScalingTerm, Translation], Float[Array, ""]]:
    """Do a single M-step with per-point diagonal uncertainties.

    Uses weighted Procrustes for the transformation and residual-based
    variance computation with uncertainty trace subtraction.

    Args:
        x: Target point set (n, d).
        y: Source point set (m, d).
        P: Matching matrix (m, n).
        tolerance: Termination tolerance.
        var: Current isotropic variance.
        unc_x: Target per-point, per-dimension variances (n, d).
        unc_y: Moving per-point, per-dimension variances (m, d).

    Returns:
        tuple[tuple[RotationMatrix, ScalingTerm, Translation], Float[Array, ""]]:
            Updated transform parameters and variance.
    """
    R, s, t = update_transform_uncertainty(x, y, P, var, unc_x, unc_y)
    y_t = transform(y, R, s, t)
    new_var = update_variance_uncertainty(x, y_t, P, tolerance, unc_x, unc_y)
    return (R, s, t), new_var
