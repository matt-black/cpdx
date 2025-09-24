import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

from ._matching import MatchingMatrix
from ._matching import expectation
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
    n, d = ref.shape
    m, _ = mov.shape
    var_i = (jnp.sum(sqdist(ref, mov)) / (m * n * d)).item()

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


def align_fixed_iter(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    outlier_prob: float,
    num_iter: int,
) -> tuple[
    tuple[MatchingMatrix, RotationMatrix, ScalingTerm, Translation],
    Float[Array, " {num_iter}"],
]:

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


def transform(
    y: Float[Array, "m d"],
    R: RotationMatrix,
    s: ScalingTerm,
    t: Translation,
) -> Float[Array, "m d"]:
    return s * (y @ R.T) + t


def maximization(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    P: Float[Array, "m n"],
    tolerance: float,
) -> tuple[tuple[RotationMatrix, ScalingTerm, Translation], Float[Array, ""]]:
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
