import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array
from jaxtyping import Float

from .._matching import MatchingMatrix
from ..deformable import KernelMatrix
from ..rigid import RotationMatrix
from ..rigid import ScalingTerm
from ..rigid import Translation
from ..util import sqdist
from .kernel import KernelFunction


__all__ = [
    "affinity_matrix",
    "initialize",
    "apply_T",
    "apply_Tinv",
    "residual",
    "interpolate",
]


def affinity_matrix(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    kernel: KernelFunction,
    beta: float,
) -> KernelMatrix:
    """Calculate the affinity matrix between pairs of points in the two points clouds, `x` and `y`.

    Args:
        x (Float[Array, "n d"]): reference point cloud
        y (Float[Array, "m d"]): moving point cloud
        kernel (KernelFunction): kernel function
        beta (float | Float[Array, ""]): shape parameter of kernel

    Returns:
        KernelMatrix
    """
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
    Float[Array, " m"],  # alpha_m (mixing coefficients)
    Float[Array, " m"],  # sigma_m
    Float[Array, ""],  # initial variance
]:
    """Compute gram matrix for moving points and determine initial parameters for the BCPD algorithm. Uses the method described in section 4.3.5 of [1].

    Args:
        x (Float[Array, "n d"]): reference point cloud
        y (Float[Array, "m d"]): moving point cloud
        kernel (KernelFunction): kernel function
        beta (float): shape parameter of kernel
        gamma (float): scalar to scale initial variance estimate by

    Returns:
        tuple[KernelMatrix, Float[Array, " m"], Float[Array, " m"], Float[Array, ""]]

    References:
        [1] O. Hirose, "A Bayesian Formulation of Coherent Point Drift," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 7, pp. 2269-2286, 1 July 2021, doi: 10.1109/TPAMI.2020.2971687.
    """
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
    """Apply rigid transform to points.

    Args:
        x (Float[Array, "n d"]): points to transform.
        R (RotationMatrix): `d`-dimensional rotation matrix
        s (ScalingTerm): scalar, isotropic scaling term
        t (Translation): translation

    Returns:
        Float[Array, "n d"]: transformed points
    """
    return (R.T @ x.T).T * s + t


def apply_Tinv(
    x: Float[Array, "n d"],
    R: RotationMatrix,
    s: ScalingTerm,
    t: Translation,
) -> Float[Array, "n d"]:
    """Apply inverse of specified rigid transform to points.

    Args:
        x (Float[Array, "n d"]): points to transform.
        R (RotationMatrix): `d`-dimensional rotation matrix
        s (ScalingTerm): scalar, isotropic scaling term
        t (Translation): translation

    Returns:
        Float[Array, "n d"]: transformed points
    """
    return apply_T(x - t[None, :], R, 1 / s, jnp.array(0.0))


def dimension_bounds(x: Float[Array, " n"]) -> Float[Array, " 2"]:
    """Get min and max values for bounding box of point cloud along some dimension.

    Args:
        x (Float[Array, " n"]): all coordinate values in some dimension

    Returns:
        Float[Array, " 2"]: [min, max] coordinate in dimension
    """
    return jnp.asarray([jnp.amin(x), jnp.amax(x)])


def residual(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    P: MatchingMatrix,
    R: RotationMatrix,
    s: ScalingTerm,
    t: Translation,
    eps: float = 1e-12,
) -> Float[Array, "m d"]:
    nu = jnp.clip(jnp.sum(P, axis=1), eps)
    ref_hat = jnp.diag(jnp.divide(1.0, nu)) @ P @ ref
    return apply_Tinv(ref_hat, R, s, t) - mov


@Partial(jax.jit, static_argnums=(5, 6, 7, 8, 9, 10))
def interpolate(
    mov: Float[Array, "m d"],
    interp: Float[Array, "i d"],
    resid: Float[Array, "m d"],
    P: Float[Array, "n m"],
    G_mm: Float[Array, "m m"],
    kernel: KernelFunction,
    beta: float,
    s: float,
    lambda_: float,
    var: float,
    eps: float = 1e-12,
) -> Float[Array, "i d"]:
    nu = jnp.clip(jnp.sum(P, axis=1), eps)
    psi = (lambda_ * var / s**2) * jnp.diag(1.0 / nu)
    G = affinity_matrix(interp, mov, kernel, beta)
    return G @ jnp.linalg.inv(G_mm + psi) @ resid
