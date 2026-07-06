"""Bayesian coherent point drift

References
---
[1] O. Hirose, "A Bayesian Formulation of Coherent Point Drift," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 7, pp. 2269-2286, 1 July 2021, doi: 10.1109/TPAMI.2020.2971687.
"""

import math
from typing import Union

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

from .._matching import MatchingMatrix
from ..nonrigid import KernelMatrix
from ..rigid import RotationMatrix
from ..rigid import ScalingTerm
from ..rigid import Translation
from ._private import update_matching
from ._private import update_matching_masked
from ._private import update_nonrigid
from ._private import update_rigid
from ._private import update_variance
from .kernel import KernelFunction
from .normalization import denormalize_parameters
from .normalization import normalize_points
from .util import initialize
from .util import transform as transform


__all__ = [
    "align",
    "align_with_ic",
    "transform",
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
    fixed_alpha: Float[Array, " m"] | None = None,
    transform_mode: str = "both",
    normalize_input: bool = False,
    debias: bool = False,
    return_weights: bool = False,
    mask: Float[Array, "m n"] | None = None,
    burn_in: int = 0,
) -> tuple[
    TransformParams,
    Union[Float[Array, " {num_iter}"], tuple[Float[Array, ""], int]],
]:
    """Align the moving points onto the reference points using bayesian coherent point drift (bcpd).

    Initial conditions are set using the `cpdx.bayes.util.initialize`, see docstrings of that function for details. For cases where you have initial conditions you'd like to use for the optimization, use `align_with_ic`.

    Args:
        ref (Float[Array, "n d"]): reference points
        mov (Float[Array, "m d"]): moving points
        outlier_prob (float): outlier probability, should be in range [0,1].
        num_iter (int): maximum # of iterations to optimize for. if tolerance is `None`, this is the number of iterations that will be optimized for.
        tolerance (float): tolerance for residual variance, below which the algorithm will terminate. If `None`, a fixed number of iterations is used.
        kernel (KernelFunction): the kernel function to use when calculating the gram matrix
        lambda_param (float): regularization parameter (usually termed "lambda" in the literature) for motion coherence.
        kernel_beta (float): shape parameter for the kernel function. For gaussian kernels, this corresponds to the standard deviation of the gaussian.
        gamma (float): scalar to scale initial variance estimate by.
        kappa (float): shape parameter for Dirichlet distribution used during matching. Set to `math.inf` if mixing coefficients for all points should be equal.
        fixed_alpha (Float[Array, " m"] | None): optional fixed mixing coefficients. If provided, alpha_m will not be updated during optimization.
        transform_mode (str): transform mode, one of "both" (default), "rigid", or "nonrigid". Controls which transform components are updated.
        normalize_input (bool): if True, inputs are normalized to zero mean and unit variance before alignment, and results are denormalized. Defaults to False.
        debias (bool): if True, includes a de-biasing term (average deformation uncertainty) in the rigid scale calculation. This can stabilize scale recovery in some cases but may cause collapse in others. Defaults to False.
        return_weights (bool): if True, returns the GP weights `W` instead of the deformation vectors `v` in the `TransformParams` tuple. Defaults to False.
        mask (Float[Array, "m n"] | None): optional mask of shape `(m, n)` where nonzero entries indicate valid
            (source_m, target_n) pairs. Zero entries are treated as impossible matches and excluded from the
            E-step. If `None` (default), all pairs are considered valid.
        burn_in (int): number of initial iterations to run with rigid-only registration
            before enabling non-rigid deformation. Only effective when `transform_mode="both"`.
            During burn-in, the algorithm estimates a good rigid alignment (R, s, t) with
            zero non-rigid displacement, then transitions to full rigid+nonrigid optimization.
            `num_iter` represents total iterations, so the second phase gets `num_iter - burn_in`
            iterations. Defaults to 0 (no burn-in).

    Returns:
        tuple[TransformParams, Union[Float[Array, " {num_iter}"], tuple[Float[Array, ""], int]]: the fitted transform parameters (the matching matrix, the rigid transform parameters, and the learned vector field) along with a tuple describing the optimization. If `tolerance=None`, a vector of variances at each step of the iteration is returned. Otherwise, the final variance and the number of iterations the algorithm was run for is returned.

    Notes:
        Unpack the return parameters like `(P, R, s, t, v), _ = align(...)`.
    """
    _, d = mov.shape

    # Validate transform_mode
    if transform_mode not in ["both", "rigid", "nonrigid"]:
        raise ValueError(
            f"transform_mode must be 'both', 'rigid', or 'nonrigid', got '{transform_mode}'"
        )

    # Validate burn_in
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}")

    # Input Normalization
    if normalize_input:
        ref_norm, mu_x, sigma_x = normalize_points(ref)
        mov_norm, mu_y, sigma_y = normalize_points(mov)
    else:
        ref_norm, mov_norm = ref, mov
        mu_x, mu_y, sigma_x, sigma_y = 0, 0, 0, 0

    # Initialize G and alpha_m once (used by both burn-in and non-burn-in paths)
    G, alpha_m, sigma_m, var_i = initialize(
        ref_norm, mov_norm, kernel, kernel_beta, gamma
    )
    if fixed_alpha is not None:
        alpha_m = fixed_alpha

    # Burn-in two-phase strategy: when burn_in > 0 and transform_mode == "both",
    # run a rigid-only phase first, then continue with full optimization.
    if burn_in > 0 and transform_mode == "both":
        if burn_in >= num_iter:
            raise ValueError(
                f"burn_in ({burn_in}) must be less than num_iter ({num_iter})"
            )

        m, _ = mov_norm.shape

        # Phase 1: Rigid-only burn-in (always fixed iterations)
        (P1, R, s, t, _), var1_result = align_with_ic(
            ref_norm, mov_norm,
            outlier_prob, burn_in, None,  # tolerance=None for fixed iterations
            lambda_param, kappa,
            G, jnp.eye(d), jnp.array(1.0), jnp.zeros(d), jnp.zeros_like(mov_norm),
            jnp.zeros_like(sigma_m),  # sigma_m=0 for rigid
            alpha_m, var_i,
            fixed_alpha=fixed_alpha,
            transform_mode="rigid",
            normalize_input=False,
            debias=debias,
            return_weights=False,
            mask=mask,
            burn_in=0,  # Prevent infinite recursion
        )
        # var1_result is Float[Array, "{burn_in}"] because tolerance=None
        var1 = var1_result  # type: ignore[assignment]

        # Phase 2: Full rigid + nonrigid from rigid results
        sigma_m_fresh = jnp.full((m,), 1e-4)
        remaining_iter = num_iter - burn_in
        var_last = jnp.array(float(var1[-1]))  # scalar variance from last burn-in step

        if tolerance is None:
            (P, R, s, t, v), var2_result = align_with_ic(
                ref_norm, mov_norm,
                outlier_prob, remaining_iter, None,
                lambda_param, kappa,
                G, R, s, t, jnp.zeros_like(mov_norm),
                sigma_m_fresh, alpha_m, var_last,
                fixed_alpha=fixed_alpha,
                transform_mode="both",
                normalize_input=False,
                debias=debias,
                return_weights=return_weights,
                mask=mask,
                burn_in=0,  # Prevent recursion
            )
            var2 = var2_result  # type: ignore[assignment]
            var = jnp.concatenate([var1, var2])  # type: ignore[arg-type]
        else:
            (P, R, s, t, v), (var_final, iters) = align_with_ic(
                ref_norm, mov_norm,
                outlier_prob, remaining_iter, tolerance,
                lambda_param, kappa,
                G, R, s, t, jnp.zeros_like(mov_norm),
                sigma_m_fresh, alpha_m, var_last,
                fixed_alpha=fixed_alpha,
                transform_mode="both",
                normalize_input=False,
                debias=debias,
                return_weights=return_weights,
                mask=mask,
                burn_in=0,  # Prevent recursion
            )
            var = (var_final, burn_in + iters)

        # Denormalize if needed
        if normalize_input:
            R, s, t, v = denormalize_parameters(
                R, s, t, v,
                mu_y,  # pyright: ignore[reportArgumentType]
                sigma_y,  # pyright: ignore[reportArgumentType]
                mu_x,  # pyright: ignore[reportArgumentType]
                sigma_x,  # pyright: ignore[reportArgumentType]
            )
        return (P, R, s, t, v), var

    # No burn-in or non-"both" mode: existing behavior
    # (G, alpha_m, sigma_m, var_i already initialized above)

    # If transform_mode is rigid, force sigma_m to 0 (no deformation uncertainty)
    if transform_mode == "rigid":
        sigma_m = jnp.zeros_like(sigma_m)

    R = jnp.eye(d)
    s = jnp.array(1.0)
    t = jnp.zeros((d,))
    v_hat = jnp.zeros_like(mov)

    if tolerance is None:
        (P, R, s, t, v), var = _align_fixed_iter(
            ref_norm,
            mov_norm,
            lambda_param,
            outlier_prob,
            kappa,
            num_iter,
            G,
            R,
            s,
            t,
            v_hat,
            sigma_m,
            alpha_m,
            var_i,
            fixed_alpha,
            transform_mode,
            debias,
            return_weights,
            mask,
        )
    else:
        (P, R, s, t, v), var = _align_tolerance(
            ref_norm,
            mov_norm,
            lambda_param,
            outlier_prob,
            kappa,
            tolerance,
            num_iter,
            G,
            R,
            s,
            t,
            v_hat,
            sigma_m,
            alpha_m,
            var_i,
            fixed_alpha,
            transform_mode,
            debias,
            return_weights,
            mask,
        )

    # Denormalize if needed
    if normalize_input:
        R, s, t, v = denormalize_parameters(
            R,
            s,
            t,
            v,
            mu_y,  # pyright: ignore[reportArgumentType]
            sigma_y,  # pyright: ignore[reportArgumentType]
            mu_x,  # pyright: ignore[reportArgumentType]
            sigma_x,  # pyright: ignore[reportArgumentType]
        )

    return (P, R, s, t, v), var


def align_with_ic(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    outlier_prob: float,
    num_iter: int,
    tolerance: float | None,
    lambda_param: float,
    kappa: float,
    # initial conditions
    G: KernelMatrix,
    R: RotationMatrix,
    s: ScalingTerm,
    t: Translation,
    v: VectorField,
    sigma_m: Float[Array, " m"],
    alpha_m: Float[Array, " m"],
    var_i: Float[Array, ""],
    # optional parameters
    fixed_alpha: Float[Array, " m"] | None = None,
    transform_mode: str = "both",
    normalize_input: bool = False,
    debias: bool = False,
    return_weights: bool = False,
    mask: Float[Array, "m n"] | None = None,
    burn_in: int = 0,
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
        tolerance (float): tolerance for residual variance, below which the algorithm will terminate. If `None`, a fixed number of iterations is used.
        lambda_param (float): regularization parameter (usually termed "lambda" in the literature) for motion coherence.
        kappa (float): shape parameter for Dirichlet distribution used during matching. Set to `math.inf` if mixing coefficients for all points should be equal.
        G (KernelMatrix): the gram matrix between all moving points
        R (RotationMatrix): initial rotation matrix
        s (ScalingTerm): initial scalar for isotropic scaling
        t (Translation): initial translation
        v (VectorField): initial vector field on nonlinear deformations
        sigma_m (Float[Array, " m"]): initial per-moving point variances
        alpha_m (Float[Array, " m"]): initial mixing coefficients
        var_i (Float[Array, ""]): initial estimate of residual variance
        fixed_alpha (Float[Array, " m"] | None): optional fixed mixing coefficients. If provided, alpha_m will not be updated during optimization.
        transform_mode (str): transform mode, one of "both" (default), "rigid", or "nonrigid". Controls which transform components are updated.
        normalize_input (bool): if True, inputs are normalized to zero mean and unit variance before alignment, and results are denormalized. Defaults to False.
        debias (bool): if True, includes a de-biasing term in the rigid scale calculation. Defaults to False.
        return_weights (bool): if True, returns the GP weights `W` instead of the deformation vectors `v` in the `TransformParams` tuple. Defaults to False.
        mask (Float[Array, "m n"] | None): optional mask of shape `(m, n)` where nonzero entries indicate valid
            (source_m, target_n) pairs. Zero entries are treated as impossible matches and excluded from the
            E-step. If `None` (default), all pairs are considered valid.
        burn_in (int): number of initial iterations to run with rigid-only registration
            before enabling non-rigid deformation. Only effective when `transform_mode="both"`.
            During burn-in, the algorithm estimates a good rigid alignment (R, s, t) with
            zero non-rigid displacement, then transitions to full rigid+nonrigid optimization.
            `num_iter` represents total iterations, so the second phase gets `num_iter - burn_in`
            iterations. Defaults to 0 (no burn-in).

    Returns:
        tuple[TransformParams, Union[Float[Array, " {num_iter}"], tuple[Float[Array, ""], int]]: the fitted transform parameters (the matching matrix, the rigid transform parameters, and the learned vector field) along with a tuple describing the optimization. If `tolerance=None`, a vector of variances at each step of the iteration is returned. Otherwise, the final variance and the number of iterations the algorithm was run for is returned.

    Notes:
        Unpack the return parameters like `(P, R, s, t, v), _ = align(...)`.
    """
    # Validate transform_mode
    if transform_mode not in ["both", "rigid", "nonrigid"]:
        raise ValueError(
            f"transform_mode must be 'both', 'rigid', or 'nonrigid', got '{transform_mode}'"
        )

    # Validate burn_in
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}")

    # Input Normalization
    if normalize_input:
        ref_norm, mu_x, sigma_x = normalize_points(ref)
        mov_norm, mu_y, sigma_y = normalize_points(mov)
    else:
        ref_norm, mov_norm = ref, mov
        mu_x, mu_y, sigma_x, sigma_y = 0, 0, 0, 0

    # Use fixed_alpha if provided
    if fixed_alpha is not None:
        alpha_m = fixed_alpha

    # Burn-in two-phase strategy
    if burn_in > 0 and transform_mode == "both":
        if burn_in >= num_iter:
            raise ValueError(
                f"burn_in ({burn_in}) must be less than num_iter ({num_iter})"
            )

        m, _ = mov_norm.shape

        # Phase 1: Rigid-only burn-in (always fixed iterations)
        (P1, R1, s1, t1, _), var1_result = align_with_ic(
            ref_norm, mov_norm,
            outlier_prob, burn_in, None,  # tolerance=None for fixed iterations
            lambda_param, kappa,
            G, R, s, t, v,
            jnp.zeros_like(sigma_m),  # sigma_m=0 for rigid
            alpha_m, var_i,
            fixed_alpha=fixed_alpha,
            transform_mode="rigid",
            normalize_input=False,
            debias=debias,
            return_weights=False,
            mask=mask,
            burn_in=0,  # Prevent recursion
        )
        var1 = var1_result  # type: ignore[assignment]

        # Phase 2: Full rigid + nonrigid from rigid results
        sigma_m_fresh = jnp.full((m,), 1e-4)
        remaining_iter = num_iter - burn_in
        var_last = jnp.array(float(var1[-1]))

        if tolerance is None:
            (P, R2, s2, t2, v2), var2_result = align_with_ic(
                ref_norm, mov_norm,
                outlier_prob, remaining_iter, None,
                lambda_param, kappa,
                G, R1, s1, t1, jnp.zeros((m, mov_norm.shape[1])),
                sigma_m_fresh, alpha_m, var_last,
                fixed_alpha=fixed_alpha,
                transform_mode="both",
                normalize_input=False,
                debias=debias,
                return_weights=return_weights,
                mask=mask,
                burn_in=0,
            )
            var2 = var2_result  # type: ignore[assignment]
            var = jnp.concatenate([var1, var2])  # type: ignore[arg-type]
            R, s, t, v = R2, s2, t2, v2
        else:
            (P, R, s, t, v), (var_final, iters) = align_with_ic(
                ref_norm, mov_norm,
                outlier_prob, remaining_iter, tolerance,
                lambda_param, kappa,
                G, R1, s1, t1, jnp.zeros((m, mov_norm.shape[1])),
                sigma_m_fresh, alpha_m, var_last,
                fixed_alpha=fixed_alpha,
                transform_mode="both",
                normalize_input=False,
                debias=debias,
                return_weights=return_weights,
                mask=mask,
                burn_in=0,
            )
            var = (var_final, burn_in + iters)

        # Denormalize if needed
        if normalize_input:
            R, s, t, v = denormalize_parameters(
                R, s, t, v,
                mu_y,  # pyright: ignore[reportArgumentType]
                sigma_y,  # pyright: ignore[reportArgumentType]
                mu_x,  # pyright: ignore[reportArgumentType]
                sigma_x,  # pyright: ignore[reportArgumentType]
            )
        return (P, R, s, t, v), var

    # No burn-in or non-"both" mode: existing behavior
    if tolerance is not None:
        (P, R, s, t, v), var = _align_tolerance(
            ref_norm,
            mov_norm,
            lambda_param,
            outlier_prob,
            kappa,
            tolerance,
            num_iter,
            G,
            R,
            s,
            t,
            v,
            sigma_m,
            alpha_m,
            var_i,
            fixed_alpha,
            transform_mode,
            debias,
            return_weights,
            mask,
        )
    else:
        (P, R, s, t, v), var = _align_fixed_iter(
            ref_norm,
            mov_norm,
            lambda_param,
            outlier_prob,
            kappa,
            num_iter,
            G,
            R,
            s,
            t,
            v,
            sigma_m,
            alpha_m,
            var_i,
            fixed_alpha,
            transform_mode,
            debias,
            return_weights,
            mask,
        )

    # Denormalize if needed
    if normalize_input:
        R, s, t, v = denormalize_parameters(
            R,
            s,
            t,
            v,
            mu_y,  # pyright: ignore[reportArgumentType]
            sigma_y,  # pyright: ignore[reportArgumentType]
            mu_x,  # pyright: ignore[reportArgumentType]
            sigma_x,  # pyright: ignore[reportArgumentType]
        )

    return (P, R, s, t, v), var


type _StateType = tuple[
    MatchingMatrix,
    RotationMatrix,
    ScalingTerm,
    Translation,
    Float[Array, " m"],  # sigma_m
    Float[Array, " m"],  # alpha_m
    Float[Array, "m d"],  # v_hat (current vector field)
    Float[Array, "m d"],  # W (current GP weights)
    Float[Array, "m d"],  # y_hat (current aligned moving points)
    Float[Array, ""],  # current variance
    int,  # current iteration
]


def _align_tolerance(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    lambda_: float,
    outlier_prob: float,
    kappa: float,
    tolerance: float,
    max_iter: int,
    G: KernelMatrix,
    R: RotationMatrix,
    s: ScalingTerm,
    t: Translation,
    v_hat: Float[Array, "m d"],
    sigma_m: Float[Array, " m"],
    alpha_m: Float[Array, " m"],
    var_i: Float[Array, ""],
    fixed_alpha: Float[Array, " m"] | None,
    transform_mode: str,
    debias: bool,
    return_weights: bool,
    mask: Float[Array, "m n"] | None,
) -> tuple[TransformParams, tuple[Float[Array, ""], int]]:
    n, d = x.shape
    m, _ = y.shape

    def cond_fun(a: _StateType) -> Bool:
        _, _, _, _, _, _, _, _, _, var, iter_num = a
        return jnp.logical_and(var > tolerance, iter_num < max_iter)

    def body_fun(a: _StateType) -> _StateType:
        # unpack
        _, R, s, t, sigma_m, alpha_m, v_hat, W, y_hat, var, iter_num = a
        # expectation step
        if mask is not None:
            P, nu, nu_prime, n_hat, x_hat = update_matching_masked(
                x, y_hat, sigma_m, alpha_m, s, var, outlier_prob, mask
            )
        else:
            P, nu, nu_prime, n_hat, x_hat = update_matching(
                x, y_hat, sigma_m, alpha_m, s, var, outlier_prob
            )

        # maximization step depends on transform_mode
        if transform_mode == "rigid":
            # Only update rigid transform, keep v_hat = 0
            u_hat = y
            # Update alpha_m for matching (if not fixed)
            if fixed_alpha is not None:
                alpha_m = fixed_alpha
            elif math.isinf(kappa):
                alpha_m = nu / n_hat
            else:
                alpha_m = jnp.exp(
                    digamma(kappa + nu) - digamma(kappa * m + n_hat)
                )

            R, s, t, _ = update_rigid(
                y, x_hat, u_hat, sigma_m, nu, n_hat, debias
            )
            var_bar = jnp.array(0.0)
            v_hat = jnp.zeros_like(y)
            W = jnp.zeros_like(y)

        elif transform_mode == "nonrigid":
            # Only update nonrigid deformation, keep R=I, s=1, t=0
            v_hat, u_hat, sigma_m, alpha_m, W = update_nonrigid(
                x_hat,
                y,
                G,
                R,
                s,
                t,
                nu,
                var,
                n_hat,
                kappa,
                lambda_,
                fixed_alpha,
            )
            R = jnp.eye(d)
            s = jnp.array(1.0)
            t = jnp.zeros(d)
            var_bar = jnp.array(0.0)

        else:  # "both"
            # Update both nonrigid and rigid (current behavior)
            v_hat, u_hat, sigma_m, alpha_m, W = update_nonrigid(
                x_hat,
                y,
                G,
                R,
                s,
                t,
                nu,
                var,
                n_hat,
                kappa,
                lambda_,
                fixed_alpha,
            )
            R, s, t, var_bar = update_rigid(
                y, x_hat, u_hat, sigma_m, nu, n_hat, debias
            )

        # remap points using updated transform
        y_hat = transform(y + v_hat, R, s, t)
        # update variance to track how well the point clouds match
        var = update_variance(x, y_hat, P, s, nu, nu_prime, n_hat, var_bar)
        return (
            P,
            R,
            s,
            t,
            sigma_m,
            alpha_m,
            v_hat,
            W,
            y_hat,
            var,
            iter_num + 1,
        )

    init_W = jnp.zeros_like(y)
    P, R, s, t, sigma_m, alpha_m, v_hat, W, _, var, iter_num = (
        jax.lax.while_loop(
            cond_fun,
            body_fun,
            (
                jnp.empty((m, n)),
                R,
                s,
                t,
                sigma_m,
                alpha_m,
                v_hat,
                init_W,
                y,
                var_i,
                0,
            ),
        )
    )
    return (P, R, s, t, jax.lax.select(return_weights, W, v_hat)), (
        var,
        iter_num,
    )


type _CarryType = tuple[
    MatchingMatrix,
    RotationMatrix,
    ScalingTerm,
    Translation,
    Float[Array, " m"],  # sigma_m
    Float[Array, " m"],  # alpha_m
    Float[Array, "m d"],  # v_hat (current vector field)
    Float[Array, "m d"],  # W (current GP weights)
    Float[Array, "m d"],  # y_hat (current aligned moving points)
    Float[Array, ""],  # current variance
]


def _align_fixed_iter(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    lambda_: float,
    outlier_prob: float,
    kappa: float,
    num_iter: int,
    G: KernelMatrix,
    R: RotationMatrix,
    s: ScalingTerm,
    t: Translation,
    v_hat: Float[Array, "m d"],
    sigma_m: Float[Array, " m"],
    alpha_m: Float[Array, " m"],
    var_i: Float[Array, ""],
    fixed_alpha: Float[Array, " m"] | None,
    transform_mode: str,
    debias: bool,
    return_weights: bool,
    mask: Float[Array, "m n"] | None,
) -> tuple[TransformParams, Float[Array, " {num_iter}"]]:
    n, d = x.shape
    m, _ = y.shape

    def scan_fun(
        carry: _CarryType,
        _,
    ) -> tuple[_CarryType, Float[Array, ""]]:
        # unpack the carry
        _, R, s, t, sigma_m, alpha_m, v_hat, W, y_hat, var = carry
        # update matching
        if mask is not None:
            P, nu, nu_prime, n_hat, x_hat = update_matching_masked(
                x, y_hat, sigma_m, alpha_m, s, var, outlier_prob, mask
            )
        else:
            P, nu, nu_prime, n_hat, x_hat = update_matching(
                x, y_hat, sigma_m, alpha_m, s, var, outlier_prob
            )

        # maximization step depends on transform_mode
        if transform_mode == "rigid":
            # Only update rigid transform, keep v_hat = 0
            u_hat = y
            # Update alpha_m for matching (if not fixed)
            if fixed_alpha is not None:
                alpha_m = fixed_alpha
            elif math.isinf(kappa):
                alpha_m = nu / n_hat
            else:
                alpha_m = jnp.exp(
                    digamma(kappa + nu) - digamma(kappa * m + n_hat)
                )
            R, s, t, _ = update_rigid(
                y, x_hat, u_hat, sigma_m, nu, n_hat, debias
            )
            var_bar = jnp.array(0.0)
            v_hat = jnp.zeros_like(y)
            W = jnp.zeros_like(y)

        elif transform_mode == "nonrigid":
            # Only update nonrigid deformation, keep R=I, s=1, t=0
            v_hat, u_hat, sigma_m, alpha_m, W = update_nonrigid(
                x_hat,
                y,
                G,
                R,
                s,
                t,
                nu,
                var,
                n_hat,
                kappa,
                lambda_,
                fixed_alpha,
            )
            R = jnp.eye(d)
            s = jnp.array(1.0)
            t = jnp.zeros(d)
            var_bar = jnp.array(0.0)

        else:  # "both"
            # Update both nonrigid and rigid (current behavior)
            v_hat, u_hat, sigma_m, alpha_m, W = update_nonrigid(
                x_hat,
                y,
                G,
                R,
                s,
                t,
                nu,
                var,
                n_hat,
                kappa,
                lambda_,
                fixed_alpha,
            )
            R, s, t, var_bar = update_rigid(
                y, x_hat, u_hat, sigma_m, nu, n_hat, debias
            )

        y_hat = transform(y + v_hat, R, s, t)
        var = update_variance(x, y_hat, P, s, nu, nu_prime, n_hat, var_bar)
        return (P, R, s, t, sigma_m, alpha_m, v_hat, W, y_hat, var), var

    init_W = jnp.zeros_like(y)
    (P, R, s, t, _, _, v_hat, W, _, _), varz = jax.lax.scan(
        scan_fun,
        (jnp.empty((m, n)), R, s, t, sigma_m, alpha_m, v_hat, init_W, y, var_i),
        None,
        length=num_iter,
    )
    return (P, R, s, t, jax.lax.select(return_weights, W, v_hat)), varz
