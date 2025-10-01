"""Bayesian coherent point drift

References
---
[1] O. Hirose, "A Bayesian Formulation of Coherent Point Drift," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 7, pp. 2269-2286, 1 July 2021, doi: 10.1109/TPAMI.2020.2971687.
"""

from typing import Union

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

from .._matching import MatchingMatrix
from ..rigid import RotationMatrix
from ..rigid import ScalingTerm
from ..rigid import Translation
from ._private import update_deformable
from ._private import update_matching
from ._private import update_rigid
from ._private import update_variance
from .kernel import KernelFunction
from .util import apply_T
from .util import initialize


__all__ = [
    "align",
    "TransformParams",
    "VectorField",
]


type VectorField = Float[Array, "m d"]


type TransformParams = tuple[
    MatchingMatrix, RotationMatrix, ScalingTerm, Translation, VectorField
]


def align(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    outlier_prob: float,
    num_iter: int,
    tolerance: float | None,
    kernel: KernelFunction,
    lambda_param: float,
    kernel_beta: float,
    gamma: float,
    kappa: float,
) -> tuple[
    TransformParams,
    Union[Float[Array, " {num_iter}"], tuple[Float[Array, ""], int]],
]:
    """Align the moving points onto the reference points using bayesian coherent point drift (bcpd).

    Args:
        ref (Float[Array, "n d"]): reference points
        mov (Float[Array, "m d"]): moving points
        outlier_prob (float): outlier probability, should be in range [0,1].
        num_iter (int): maximum # of iterations to optimize for. if tolerance is `None`, this is the number of iterations that will be optimized for.
        tolerance (float): tolerance for matching variance, below which the algorithm will terminate. If `None`, a fixed number of iterations is used.
        kernel (KernelFunction):
        lambda_param (float): regularization parameter (usually termed "lambda" in the literature) for motion coherence.
        kernel_beta (float): shape parameter for the kernel function. For gaussian kernels, this corresponds to the standard deviation of the gaussian.
        gamma (float): scalar to scale initial variance estimate by.
        kappa (float): shape parameter for Dirichlet distribution used during matching. Set to `math.inf` if mixing coefficients for all points should be equal.

    Returns:
        tuple[TransformParams, Union[Float[Array, " {num_iter}"], tuple[Float[Array, ""], int]]: the fitted transform parameters (the matching matrix, the rigid transform parameters, and the learned vector field) along with a tuple describing the optimization. If `tolerance=None`, a vector of variances at each step of the iteration is returned. Otherwise, the final variance and the number of iterations the algorithm was run for is returned.

    Notes:

        Unpack the return parameters like `(P, R, s, t, v), _ = align(...)`.
    """
    if tolerance is None:
        return _align_fixed_iter(
            ref,
            mov,
            kernel,
            kernel_beta,
            gamma,
            lambda_param,
            outlier_prob,
            kappa,
            num_iter,
        )
    else:
        return _align_tolerance(
            ref,
            mov,
            kernel,
            kernel_beta,
            gamma,
            lambda_param,
            outlier_prob,
            kappa,
            tolerance,
            num_iter,
        )


type _StateType = tuple[
    MatchingMatrix,
    RotationMatrix,
    ScalingTerm,
    Translation,
    Float[Array, " m"],  # sigma_m
    Float[Array, " m"],  # alpha_m
    Float[Array, "m d"],  # v_hat (current vector field)
    Float[Array, "m d"],  # y_hat (current aligned moving points)
    Float[Array, ""],  # current variance
    int,  # current iteration
]


def _align_tolerance(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    kernel: KernelFunction,
    beta: float,
    gamma: float,
    lambda_: float,
    outlier_prob: float,
    kappa: float,
    tolerance: float,
    max_iter: int,
) -> tuple[TransformParams, tuple[Float[Array, ""], int]]:
    n, _ = x.shape
    m, d = y.shape
    G, alpha_m, sigma_m, var_i = initialize(x, y, kernel, beta, gamma)
    # initialize transform as identity, no shift or scaling
    R = jnp.eye(d)
    s = jnp.array(1.0)
    t = jnp.zeros((d,))
    v_hat = jnp.zeros_like(y)

    def cond_fun(a: _StateType) -> Bool:
        _, _, _, _, _, _, _, _, var, iter_num = a
        return jnp.logical_and(var > tolerance, iter_num < max_iter)

    def body_fun(a: _StateType) -> _StateType:
        # unpack
        _, R, s, t, sigma_m, alpha_m, v_hat, y_hat, var, iter_num = a
        # expectation step
        P, nu, nu_prime, n_hat, x_hat = update_matching(
            x, y_hat, sigma_m, alpha_m, s, var, outlier_prob
        )
        # maximization step is update_deformable then update_rigid
        v_hat, u_hat, sigma_m, alpha_m = update_deformable(
            x_hat,
            y,
            G,
            R,
            s,
            t,
            v_hat,
            nu,
            var,
            n_hat,
            kappa,
            lambda_,
        )
        R, s, t, var_bar = update_rigid(y, x_hat, u_hat, sigma_m, nu, n_hat)
        # remap points using updated transform
        y_hat = apply_T(y + v_hat, R, s, t)
        # update variance to track how well the point clouds match
        var = update_variance(x, y_hat, P, s, nu, nu_prime, n_hat, var_bar)
        return P, R, s, t, sigma_m, alpha_m, v_hat, y_hat, var, iter_num + 1

    P, R, s, t, sigma_m, alpha_m, v_hat, _, var, iter_num = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (jnp.empty((m, n)), R, s, t, sigma_m, alpha_m, v_hat, y, var_i, 0),
    )
    return (P, R, s, t, v_hat), (var, iter_num)


type _CarryType = tuple[
    MatchingMatrix,
    RotationMatrix,
    ScalingTerm,
    Translation,
    Float[Array, " m"],  # sigma_m
    Float[Array, " m"],  # alpha_m
    Float[Array, "m d"],  # v_hat (current vector field)
    Float[Array, "m d"],  # y_hat (current aligned moving points)
    Float[Array, ""],  # current variance
]


def _align_fixed_iter(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    kernel: KernelFunction,
    beta: float,
    gamma: float,
    lambda_: float,
    outlier_prob: float,
    kappa: float,
    num_iter: int,
) -> tuple[TransformParams, Float[Array, " {num_iter}"]]:
    n, _ = x.shape
    m, d = y.shape
    G, alpha_m, sigma_m, var_i = initialize(x, y, kernel, beta, gamma)
    R = jnp.eye(d)
    s = jnp.array(1.0)
    t = jnp.zeros((d,))
    v_hat = jnp.zeros_like(y)

    def scan_fun(
        carry: _CarryType,
        _,
    ) -> tuple[_CarryType, Float[Array, ""]]:
        # unpack the carry
        _, R, s, t, sigma_m, alpha_m, v_hat, y_hat, var = carry
        # update matching
        P, nu, nu_prime, n_hat, x_hat = update_matching(
            x, y_hat, sigma_m, alpha_m, s, var, outlier_prob
        )
        v_hat, u_hat, sigma_m, alpha_m = update_deformable(
            x_hat,
            y,
            G,
            R,
            s,
            t,
            v_hat,
            nu,
            var,
            n_hat,
            kappa,
            lambda_,
        )
        R, s, t, var_bar = update_rigid(y, x_hat, u_hat, sigma_m, nu, n_hat)
        y_hat = apply_T(y + v_hat, R, s, t)
        var = update_variance(x, y_hat, P, s, nu, nu_prime, n_hat, var_bar)
        return (P, R, s, t, sigma_m, alpha_m, v_hat, y_hat, var), var

    (P, R, s, t, _, _, v_hat, _, _), varz = jax.lax.scan(
        scan_fun,
        (jnp.empty((m, n)), R, s, t, sigma_m, alpha_m, v_hat, y, var_i),
        None,
        length=num_iter,
    )
    return (P, R, s, t, v_hat), varz
