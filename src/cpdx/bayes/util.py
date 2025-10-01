import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array
from jaxtyping import Float

from ..deformable import KernelMatrix
from ..rigid import RotationMatrix
from ..rigid import ScalingTerm
from ..rigid import Translation
from ..util import sqdist
from .kernel import KernelFunction


def affinity_matrix(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    kernel: KernelFunction,
    beta: float | Float[Array, ""],
) -> KernelMatrix:
    return jax.vmap(lambda y1: jax.vmap(lambda x1: kernel(x1, y1, beta))(x))(y)


@Partial(jax.jit, static_argnums=(2, 3, 4))
def initialize(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    kernel: KernelFunction,
    beta: float,
    gamma: float,
) -> tuple[
    KernelMatrix,
    Float[Array, " m"],
    Float[Array, " m"],
    Float[Array, ""],
]:
    m, _ = y.shape
    d_t = sqdist(x, y).T
    alpha_m = jnp.ones((m,)) / m
    var = gamma * jnp.mean(d_t)
    G = jnp.clip(affinity_matrix(y, y, kernel, beta), jnp.finfo(x.dtype).eps)
    sigma_m = jnp.ones_like(alpha_m)
    return G, alpha_m, sigma_m, var


def apply_T(
    x: Float[Array, "n d"],
    R: RotationMatrix,
    s: ScalingTerm,
    t: Float[Array, " d"],
) -> Float[Array, "n d"]:
    return (R.T @ x.T).T * s + t


def apply_Tinv(
    x: Float[Array, "n d"],
    R: RotationMatrix,
    s: ScalingTerm,
    t: Translation,
) -> Float[Array, "n d"]:
    return apply_T(x - t[None, :], R, 1 / s, jnp.array(0.0))


def dimension_bounds(x: Float[Array, " n"]) -> Float[Array, " 2"]:
    return jnp.asarray([jnp.amin(x), jnp.amax(x)])
