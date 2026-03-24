import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

from ._matching import MatchingMatrix
from ._matching import expectation
from ._matching import expectation_weighted
from .util import sqdist


__all__ = [
    "AffineMatrix",
    "Translation",
    "TransformParams",
    "align",
    "align_weighted",
    "align_fixed_iter",
    "align_fixed_iter_weighted",
    "transform",
    "maximization",
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
) -> tuple[TransformParams, tuple[Float[Array, ""], int]]:
    """Align the moving points onto the reference points by affine transform.

    Args:
        ref (Float[Array, "n d"]): reference points
        mov (Float[Array, "m d"]): moving points
        outlier_prob (float): outlier probability, should be in range [0,1].
        max_iter (int): maximum # of iterations to optimize for.
        tolerance (float): tolerance for matching variance, below which the algorithm will terminate.
        moving_weights (Float[Array, " m"] | None): optional per-point weights for source points (arbitrary positive values). If None, uniform weights are used.
        mask (Float[Array, "m n"] | None): optional mask matrix where nonzero entries indicate valid matches.

    Returns:
        tuple[TransformParams, tuple[Float[Array, ""], int]]: the fitted transform parameters (the matching matrix, affine matrix, and translation) along with the final variance and the number of iterations that the algorithm was run for.
    """
    # initialize variance
    n, d = ref.shape
    m, _ = mov.shape
    var_i = jnp.sum(sqdist(ref, mov)) / (m * n * d)

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
        p = expectation(ref, mov_t, var, outlier_prob)
        (A, t), new_var = maximization(ref, mov, p, tolerance)
        return (A, t, P), (new_var, iter_num + 1)

    (A, t, P), (var_f, num_iter) = jax.lax.while_loop(
        cond_fun,
        body_fun,
        ((jnp.eye(d), jnp.zeros((d,)), jnp.zeros((m, n))), (var_i, 0)),
    )

    return (P, A, t), (var_f, num_iter)


def align_weighted(
    ref: Float[Array, "n d"],
    alpha_n: Float[Array, " n"],
    mov: Float[Array, "m d"],
    pi_m: Float[Array, " m"],
    outlier_prob: float,
    max_iter: int,
    tolerance: float,
) -> tuple[TransformParams, tuple[Float[Array, ""], int]]:
    # initialize variance
    n, d = ref.shape
    m, _ = mov.shape
    var_i = jnp.sum(sqdist(ref, mov)) / (m * n * d)

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
        p = expectation_weighted(ref, mov_t, var, outlier_prob, pi_m)
        (A, t), new_var = maximization_weighted(ref, mov, p, tolerance, alpha_n)
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
) -> tuple[TransformParams, Float[Array, " {num_iter}"]]:
    """Align the moving points onto the reference points by affine transform.

    Args:
        ref (Float[Array, "n d"]): reference points
        mov (Float[Array, "m d"]): moving points
        outlier_prob (float): outlier probability, should be in range [0,1].
        num_iter (int): # of iterations to optimize for.
        moving_weights (Float[Array, " m"] | None): optional per-point weights for source points (arbitrary positive values). If None, uniform weights are used.
        mask (Float[Array, "m n"] | None): optional mask matrix where nonzero entries indicate valid matches.

    Returns:
        tuple[TransformParams, Float[Array, " {num_iter}"]]: the fitted transform parameters (the matching matrix, affine matrix, and translation) along with the variance at each step of the optimization.
    """
    # initialize variance
    n, d = ref.shape
    m, _ = mov.shape
    var_i = (jnp.sum(sqdist(ref, mov)) / (m * n * d)).item()

    def scan_funa(
        a: tuple[
            tuple[AffineMatrix, Translation, MatchingMatrix],
            Float[Array, ""],
        ],
        _,
    ):
        (A, t, P), var = a
        mov_t = transform(mov, A, t)
        P = expectation(ref, mov_t, var, outlier_prob)
        (A, t), new_var = maximization(ref, mov, P, 0.0)
        return ((A, t, P), new_var), new_var

    ((A, t, P), _), varz = jax.lax.scan(
        scan_funa,
        ((jnp.eye(d), jnp.zeros((d,)), jnp.zeros((m, n))), var_i),
        None,
        length=num_iter,
    )

    return (P, A, t), varz


def align_fixed_iter_weighted(
    ref: Float[Array, "n d"],
    alpha_n: Float[Array, " n"],
    mov: Float[Array, "m d"],
    pi_m: Float[Array, " m"],
    outlier_prob: float,
    num_iter: int,
) -> tuple[TransformParams, Float[Array, " {num_iter}"]]:
    """Align the moving points onto the reference points by affine transform.

    Args:
        ref (Float[Array, "n d"]): reference points
        alpha_n (Float[Array, " n"]): reference point weights
        mov (Float[Array, "m d"]): moving points
        pi_m (Float[Array, " m"]): moving point weights
        outlier_prob (float): outlier probability, should be in range [0,1].
        num_iter (int): # of iterations to optimize for.

    Returns:
        tuple[TransformParams, Float[Array, " {num_iter}"]]: the fitted transform parameters (the matching matrix, affine matrix, and translation) along with the variance at each step of the optimization.
    """
    # initialize variance
    n, d = ref.shape
    m, _ = mov.shape
    var_i = (jnp.sum(sqdist(ref, mov)) / (m * n * d)).item()

    def scan_funa(
        a: tuple[
            tuple[AffineMatrix, Translation, MatchingMatrix],
            Float[Array, ""],
        ],
        _,
    ):
        (A, t, P), var = a
        mov_t = transform(mov, A, t)
        P = expectation_weighted(ref, mov_t, var, outlier_prob, pi_m)
        (A, t), new_var = maximization_weighted(ref, mov, P, 0.0, alpha_n)
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
    _, d = x.shape
    # update transform
    N = jnp.sum(P)
    Pt1, P1 = jnp.sum(P, axis=0), jnp.sum(P, axis=1)
    mu_x = jnp.divide(x.T @ Pt1, N)
    mu_y = jnp.divide(y.T @ P1, N)
    x_hat, y_hat = x - mu_x, y - mu_y
    B = jnp.dot(jnp.dot(x_hat.T, P.T), y_hat)
    ypy = jnp.dot(jnp.dot(y_hat.T, jnp.diag(P1)), y_hat)
    A = jnp.linalg.solve(ypy.T, B.T)
    t = mu_x.T - A.T @ mu_y.T
    # update variance
    val = jnp.trace(x_hat.T @ jnp.diag(Pt1) @ x_hat) - jnp.trace(
        x_hat.T @ P.T @ y_hat @ A.T
    )
    new = jnp.divide(val, N * d)
    var = jax.lax.select(new > 0, new, tolerance - 2 * jnp.finfo(x.dtype).eps)
    return (A, t), var


def maximization_weighted(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    P: MatchingMatrix,
    tolerance: float,
    alpha_n: Float[Array, " n"],
) -> tuple[tuple[AffineMatrix, Translation], Float[Array, ""]]:
    _, d = x.shape
    # update transform parameters
    Pt1, P1 = alpha_n * jnp.sum(P, axis=0), P @ alpha_n
    N_p = jnp.sum(alpha_n * Pt1)
    mu_x = jnp.divide(x.T @ Pt1, N_p)
    mu_y = jnp.divide(y.T @ P1, jnp.sum(P1))
    x_hat, y_hat = x - mu_x, y - mu_y
    xpy = x_hat.T @ jnp.diag(alpha_n) @ P.T @ y_hat.T
    ypy = y_hat.T @ jnp.diag(P @ alpha_n) @ y_hat
    A = jnp.linalg.solve(ypy.T, xpy.T)
    t = mu_x.T - A.T @ mu_y.T
    # update variance
    val = (
        jnp.trace(x_hat.T @ Pt1 @ x_hat)
        - 2 * jnp.trace(x_hat.T @ jnp.diag(alpha_n) @ P.T @ y_hat @ A.T)
        + jnp.trace(y_hat.T @ A.T @ jnp.diag(P1) @ A @ y_hat)
    )
    new = jnp.divide(val, N_p * d)
    var = jax.lax.select(new > 0, new, tolerance - 2 * jnp.finfo(x.dtype).eps)
    return (A, t), var
