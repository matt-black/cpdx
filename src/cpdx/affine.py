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
    "AffineMatrix",
    "Translation",
    "TransformParams",
    "align",
    "align_fixed_iter",
    "transform",
    "maximization",
    "maximization_uncertainty",
]


type AffineMatrix = Float[Array, "d d"]
type Translation = Float[Array, " d"]
type TransformParams = tuple[MatchingMatrix, AffineMatrix, Translation]


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
) -> tuple[TransformParams, tuple[Float[Array, ""], int]]:
    """Align the moving points onto the reference points by affine transform.

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

    Returns:
        tuple[TransformParams, tuple[Float[Array, ""], int]]: the fitted transform parameters (the matching matrix, affine matrix, and translation) along with the final variance and the number of iterations that the algorithm was run for.
    """
    # initialize variance
    n, d = ref.shape
    m, _ = mov.shape
    var_i = jnp.sum(sqdist(ref, mov)) / (m * n * d)
    has_unc = unc_ref is not None and unc_mov is not None

    def cond_fun(
        a: tuple[
            tuple[AffineMatrix, Translation, MatchingMatrix],
            tuple[Float[Array, ""], int],
        ],
    ) -> Bool:
        _, (var, iter_num) = a
        return jnp.logical_and(var > tolerance, iter_num < max_iter)

    def body_fun(
        a: tuple[
            tuple[AffineMatrix, Translation, MatchingMatrix],
            tuple[Float[Array, ""], int],
        ],
    ) -> tuple[
        tuple[AffineMatrix, Translation, MatchingMatrix],
        tuple[Float[Array, ""], int],
    ]:
        (A, t, P), (var, iter_num) = a
        mov_t = transform(mov, A, t)
        if has_unc:
            assert unc_ref is not None and unc_mov is not None
            P = expectation(
                ref, mov_t, var, outlier_prob,
                moving_weights=moving_weights, mask=mask,
                unc_x=unc_ref, unc_y=unc_mov,
            )
            (A, t), new_var = maximization_uncertainty(
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
            (A, t), new_var = maximization(ref, mov, P, tolerance)
        return (A, t, P), (new_var, iter_num + 1)

    (A, t, P), (var_f, num_iter) = jax.lax.while_loop(
        cond_fun,
        body_fun,
        ((jnp.eye(d), jnp.zeros((d,)), jnp.zeros((m, n))), (var_i, 0)),
    )

    return (P, A, t), (var_f, num_iter)


def align_fixed_iter(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    outlier_prob: float,
    num_iter: int,
    unc_ref: Float[Array, "n d"] | None = None,
    unc_mov: Float[Array, "m d"] | None = None,
    moving_weights: Float[Array, " m"] | None = None,
    mask: Float[Array, "m n"] | None = None,
) -> tuple[TransformParams, Float[Array, " {num_iter}"]]:
    """Align the moving points onto the reference points by affine transform.

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

    Returns:
        tuple[TransformParams, Float[Array, " {num_iter}"]]: the fitted transform parameters (the matching matrix, affine matrix, and translation) along with the variance at each step of the optimization.
    """
    # initialize variance
    n, d = ref.shape
    m, _ = mov.shape
    var_i = (jnp.sum(sqdist(ref, mov)) / (m * n * d)).item()
    has_unc = unc_ref is not None and unc_mov is not None

    def scan_funa(
        a: tuple[
            tuple[AffineMatrix, Translation, MatchingMatrix],
            Float[Array, ""],
        ],
        _,
    ):
        (A, t, P), var = a
        mov_t = transform(mov, A, t)
        if has_unc:
            assert unc_ref is not None and unc_mov is not None
            P = expectation(
                ref, mov_t, var, outlier_prob,
                moving_weights=moving_weights, mask=mask,
                unc_x=unc_ref, unc_y=unc_mov,
            )
            (A, t), new_var = maximization_uncertainty(
                ref, mov, P, 1e-6, var, unc_ref, unc_mov
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
            (A, t), new_var = maximization(ref, mov, P, 1e-6)
        return ((A, t, P), new_var), new_var

    ((A, t, P), _), varz = jax.lax.scan(
        scan_funa,
        ((jnp.eye(d), jnp.zeros((d,)), jnp.zeros((m, n))), var_i),
        None,
        length=num_iter,
    )

    return (P, A, t), varz


def transform(
    y: Float[Array, "m d"], A: Float[Array, "d d"], t: Float[Array, " d"]
) -> Float[Array, "m d"]:
    """Transform the input points by affine transform.

    Args:
        y (Float[Array, "m d"]): `d`-dimensional points to be transformed
        A (Float[Array, "d d"]): `d`-dimensional affine transform matrix
        t (Float[Array, " d"]): translation

    Returns:
        Float[Array, "m d"]: transformed points, `y @ A + t`
    """
    return y @ A + t[None, :]


def maximization(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    P: MatchingMatrix,
    tolerance: float,
) -> tuple[tuple[AffineMatrix, Translation], Float[Array, ""]]:
    """Do a single M-step.

    Args:
        x (Float[Array, "n d"]): target point set
        y (Float[Array, "m d"]): source point set
        P (MatchingMatrix): matching matrix
        tolerance (float): termination tolerance

    Returns:
        tuple[tuple[AffineMatrix, Translation], Float[Array, ""]]: updated transform parameters, and variance.
    """
    A, t = update_transform(x, y, P)
    y_t = transform(y, A, t)
    var = update_variance(x, y_t, P, tolerance)
    return (A, t), var


def update_transform(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    P: Float[Array, "m n"],
) -> tuple[Float[Array, "d d"], Float[Array, " d"]]:
    N = jnp.sum(P)
    Pt1, P1 = jnp.sum(P, axis=0), jnp.sum(P, axis=1)
    mu_x = jnp.divide(x.T @ Pt1, N)
    mu_y = jnp.divide(y.T @ P1, N)
    x_hat, y_hat = x - mu_x, y - mu_y
    B = jnp.dot(jnp.dot(x_hat.T, P.T), y_hat)
    ypy = jnp.dot(jnp.dot(y_hat.T, jnp.diag(P1)), y_hat)
    A = jnp.linalg.solve(ypy.T, B.T)
    t = mu_x.T - A.T @ mu_y.T
    return (A, t)


def update_variance(
    x: Float[Array, "n d"],
    y_t: Float[Array, "m d"],
    P: MatchingMatrix,
    tolerance: float,
) -> Float[Array, ""]:
    _, d = x.shape
    N = jnp.sum(P)
    Pt1, P1 = jnp.sum(P, axis=0), jnp.sum(P, axis=1)
    val = (
        jnp.trace(x.T @ jnp.diag(Pt1) @ x)
        - 2 * jnp.trace((P @ x).T @ y_t)
        + jnp.trace(y_t.T @ jnp.diag(P1) @ y_t)
    )
    new = jnp.divide(val, N * d)
    return jax.lax.select(new > 0, new, tolerance - 2 * jnp.finfo(x.dtype).eps)


def update_transform_uncertainty(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    P: Float[Array, "m n"],
    var: Float[Array, ""],
    unc_x: Float[Array, "n d"],
    unc_y: Float[Array, "m d"],
) -> tuple[AffineMatrix, Translation]:
    """Per-dimension weighted affine regression with uncertainty.

    For each output dimension k, solves a weighted linear regression with
    effective weights w_{mn}^{(k)} = P_{mn} / (var + unc_x[n,k] + unc_y[m,k]).

    Uses batched einsum to solve all dimensions at once.

    Args:
        x: Target point set (n, d).
        y: Source point set (m, d).
        P: Matching matrix (m, n).
        var: Current isotropic variance.
        unc_x: Target per-point, per-dimension variances (n, d).
        unc_y: Moving per-point, per-dimension variances (m, d).

    Returns:
        tuple[AffineMatrix, Translation]: affine matrix A and translation t.
    """
    _, d = x.shape

    # Per-dimension covariance: cov[m, n, k] = var + unc_x[n, k] + unc_y[m, k]
    cov_diag = var + unc_x[None, :, :] + unc_y[:, None, :]  # (m, n, d)
    # Weighted matching: w[m, n, k] = P[m, n] / cov[m, n, k]
    w_mnk = P[:, :, None] / cov_diag  # (m, n, d)

    # Aggregate weights per dimension
    P1 = jnp.sum(w_mnk, axis=1)   # (m, d) — sum over n
    Pt1 = jnp.sum(w_mnk, axis=0)  # (n, d) — sum over m
    N = jnp.sum(P1, axis=0)        # (d,) — total weight per dim

    # Per-dimension centroids
    mu_x = jnp.sum(x * Pt1, axis=0) / N                          # (d,)
    mu_y = jnp.einsum('mk,mj->kj', P1, y) / N[None, :]           # (d, d)

    # Centered coordinates
    x_hat = x - mu_x[None, :]                           # (n, d)
    y_hat_k = y[None, :, :] - mu_y[:, None, :]          # (d, m, d)

    # B[k, j] = sum_{m,n} w_{mn}^{(k)} * x_hat_{n,k} * y_hat^{(k)}_{m,j}
    r = jnp.einsum('mnk,nk->mk', w_mnk, x_hat)         # (m, d)
    B = jnp.einsum('mk,kmj->kj', r, y_hat_k)            # (d, d)

    # ypy[k, j, l] = sum_m P1[m,k] * y_hat^{(k)}_{m,j} * y_hat^{(k)}_{m,l}
    ypy = jnp.einsum('mk,kmj,kml->kjl', P1, y_hat_k, y_hat_k)  # (d, d, d)

    # Solve: for each k, ypy[k].T @ A[:,k] = B[k,:]
    A_cols = jax.vmap(lambda yk, bk: jnp.linalg.solve(yk.T, bk))(ypy, B)
    A = A_cols.T  # (d, d)

    # Translation: t[k] = mu_x[k] - A[k,:] @ mu_y[k,:]
    t = mu_x - jnp.einsum('kj,kj->k', A.T, mu_y)  # (d,)
    return A, t


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
) -> tuple[tuple[AffineMatrix, Translation], Float[Array, ""]]:
    """Do a single M-step with per-point diagonal uncertainties.

    Uses per-dimension weighted affine regression for the transformation
    and residual-based variance computation with uncertainty trace subtraction.

    Args:
        x: Target point set (n, d).
        y: Source point set (m, d).
        P: Matching matrix (m, n).
        tolerance: Termination tolerance.
        var: Current isotropic variance.
        unc_x: Target per-point, per-dimension variances (n, d).
        unc_y: Moving per-point, per-dimension variances (m, d).

    Returns:
        tuple[tuple[AffineMatrix, Translation], Float[Array, ""]]:
            Updated transform parameters and variance.
    """
    A, t = update_transform_uncertainty(x, y, P, var, unc_x, unc_y)
    y_t = transform(y, A, t)
    new_var = update_variance_uncertainty(x, y_t, P, tolerance, unc_x, unc_y)
    return (A, t), new_var
