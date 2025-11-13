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
    "RotationMatrix",
    "ScalingTerm",
    "Translation",
    "TransformParams",
    "align",
    "align_weighted",
    "align_fixed_iter",
    "align_fixed_iter_weighted",
    "transform",
    "maximization",
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

    Returns:
        tuple[TransformParams, tuple[Float[Array, ""], int]]: the fitted transform parameters (a rotation matrix, scaling term, and translation) along with the final variance and the number of iterations that the algorithm was run for.
    """
    n, d = ref.shape
    m, _ = mov.shape
    var_i = jnp.sum(sqdist(ref, mov)) / (m * n * d)

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
        P = expectation(ref, mov_t, var, outlier_prob)
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


def align_weighted(
    ref: Float[Array, "n d"],
    alpha_n: Float[Array, " n"],
    mov: Float[Array, "m d"],
    pi_m: Float[Array, " m"],
    outlier_prob: float,
    max_iter: int,
    tolerance: float,
) -> tuple[
    tuple[MatchingMatrix, RotationMatrix, ScalingTerm, Translation],
    tuple[Float[Array, ""], int],
]:
    """Align the moving points onto the reference points by rigid transform.

    Args:
        ref (Float[Array, "n d"]): reference points
        alpha_n (Float[Array, " n"]): reference point weights
        mov (Float[Array, "m d"]): moving points
        pi_m (Float[Array, " m"]): moving point weights
        outlier_prob (float): outlier probability, should be in range [0,1].
        max_iter (int): maximum # of iterations to optimize for.
        tolerance (float): tolerance for matching variance, below which the algorithm will terminate.

    Returns:
        tuple[TransformParams, tuple[Float[Array, ""], int]]: the fitted transform parameters (a rotation matrix, scaling term, and translation) along with the final variance and the number of iterations that the algorithm was run for.
    """
    n, d = ref.shape
    m, _ = mov.shape
    var_i = jnp.sum(sqdist(ref, mov)) / (m * n * d)

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
        P = expectation_weighted(ref, mov_t, var, outlier_prob, pi_m)
        (R, s, t), new_var = maximization_weighted(
            ref, mov, P, tolerance, alpha_n
        )
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

    Returns:
        tuple[TransformParams, Float[Array, " {num_iter}"]]: the fitted transform parameters (a rotation matrix, scaling term, and translation) along with the variance at each step of the optimization.
    """
    # initialize variance
    n, d = ref.shape
    m, _ = mov.shape
    var_i = (jnp.sum(sqdist(ref, mov)) / (m * n * d)).item()

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
        p = expectation(ref, mov_t, var, outlier_prob)
        (R, s, t), new_var = maximization(ref, mov, p, 0.0)
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


def align_fixed_iter_weighted(
    ref: Float[Array, "n d"],
    alpha_n: Float[Array, " n"],
    mov: Float[Array, "m d"],
    pi_m: Float[Array, " m"],
    outlier_prob: float,
    num_iter: int,
) -> tuple[
    tuple[MatchingMatrix, RotationMatrix, ScalingTerm, Translation],
    Float[Array, " {num_iter}"],
]:
    """Align the moving points onto the reference points by rigid transform.

    Args:
        ref (Float[Array, "n d"]): reference points
        alpha_n (Float[Array, " n"]): reference point weights
        mov (Float[Array, "m d"]): moving points
        pi_m (Float[Array, " m"]): moving point weights
        outlier_prob (float): outlier probability, should be in range [0,1].
        max_iter (int): maximum # of iterations to optimize for.
        tolerance (float): tolerance for matching variance, below which the algorithm will terminate.

    Returns:
        tuple[TransformParams, Float[Array, " {num_iter}"]]: the fitted transform parameters (a rotation matrix, scaling term, and translation) along with the variance at each step of the optimization.
    """
    # initialize variance
    n, d = ref.shape
    m, _ = mov.shape
    var_i = (jnp.sum(sqdist(ref, mov)) / (m * n * d)).item()

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
        p = expectation_weighted(ref, mov_t, var, outlier_prob, pi_m)
        (R, s, t), new_var = maximization_weighted(ref, mov, p, 0.0, alpha_n)
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


def maximization_weighted(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    P: Float[Array, "m n"],
    tolerance: float,
    alpha_n: Float[Array, " n"],
) -> tuple[tuple[RotationMatrix, ScalingTerm, Translation], Float[Array, ""]]:
    _, d = x.shape
    # update transform parameters
    Pt1, P1 = alpha_n * jnp.sum(P, axis=0), P @ alpha_n
    N_p = jnp.sum(alpha_n * Pt1)
    mu_x = jnp.divide(x.T @ Pt1, N_p)
    mu_y = jnp.divide(y.T @ P1, jnp.sum(P1))
    x_hat, y_hat = x - mu_x, y - mu_y
    A = x_hat.T @ jnp.diag(alpha_n) @ P.T @ y_hat
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
    numerator = (
        jnp.trace(x_hat.T @ jnp.diag(Pt1) @ x_hat)
        - 2 * s * jnp.trace(A.T @ R)
        + jnp.square(s) * jnp.trace(y_hat.T @ jnp.diag(P1) @ y_hat)
    )
    var = jnp.divide(numerator, N_p * d)
    return (R, s, t), jax.lax.select(var > 0, var, tolerance / 10.0)
