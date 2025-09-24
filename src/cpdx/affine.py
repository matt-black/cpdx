import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

from ._matching import MatchingMatrix
from ._matching import expectation
from .util import sqdist


__all__ = [
    "AffineMatrix",
    "Translation",
    "TransformParams",
    "align",
    "align_fixed_iter",
    "transform",
    "maximization",
]


type AffineMatrix = Float[Array, "d d"]
type Translation = Float[Array, " d"]
type TransformParams = tuple[AffineMatrix, Translation]


def align(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    outlier_prob: float,
    max_iter: int,
    tolerance: float,
):
    # initialize variance
    n, d = ref.shape
    m, _ = mov.shape
    var_i = (jnp.sum(sqdist(ref, mov)) / (m * n * d)).item()

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


def align_fixed_iter(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    outlier_prob: float,
    num_iter: int,
) -> tuple[
    tuple[MatchingMatrix, AffineMatrix, Translation],
    Float[Array, " {num_iter}"],
]:
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


def transform(
    y: Float[Array, "m d"], A: Float[Array, "d d"], t: Float[Array, " d"]
) -> Float[Array, "m d"]:
    return y @ A + t[None, :]


def maximization(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    P: Float[Array, "m n"],
    tolerance: float,
) -> tuple[tuple[Float[Array, "d d"], Float[Array, " d"]], Float[Array, ""]]:
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
