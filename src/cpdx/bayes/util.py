import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

from .._matching import MatchingMatrix
from ..nonrigid import KernelMatrix
from ..rigid import RotationMatrix
from ..rigid import ScalingTerm
from ..rigid import Translation
from ..util import sqdist
from .kernel import KernelFunction


__all__ = [
    "affinity_matrix",
    "initialize",
    "transform",
    "transform_inverse",
    "residual",
    "interpolate",
    "interpolate_variance",
    "interpolate_covariance",
    "invert_gp_bcpd_mapping",
    "interpolate_variance_inverse",
]

type InverseCovariance = Float[Array, "n d d"]


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
    return jax.vmap(lambda x1: jax.vmap(lambda y1: kernel(x1, y1, beta))(y))(x)


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
    sigma_m = jnp.full_like(alpha_m, 1e-4)
    return G, alpha_m, sigma_m, var


def transform(
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

    Notes:
        To apply the full BCPD transform, add the `VectorField` output to the points to transform before passing them into this function.
    """
    return s * (x @ R.T) + t


def transform_inverse(
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
    return transform(x - t[None, :], R.T, 1 / s, jnp.array(0.0))


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
    return transform_inverse(ref_hat, R, s, t) - mov


def interpolate(
    mov: Float[Array, "m d"],
    interp: Float[Array, "i d"],
    resid: Float[Array, "m d"],
    P: Float[Array, "m n"],
    G_mm: Float[Array, "m m"],
    kernel: KernelFunction,
    beta: float,
    s: float,
    lambda_: float,
    var: float,
    eps: float = 1e-12,
) -> Float[Array, "i d"]:
    """Predict the mean vector at each interpolation point.

    Args:
        mov (Float[Array, "m d"]): points from moving point cloud
        interp (Float[Array, "i d"]): points to interpolate vectors at
        resid (Float[Array, "m d"]): fitting residual
        P (Float[Array, "n m"]): fitted matching matrix
        G_mm (Float[Array, "m m"]): gram matrix between all pairs of points in moving point cloud
        kernel (KernelFunction): kernel function
        beta (float): shape parameter for kernel
        s (float): fitted scaling term
        lambda_ (float): regularization parameter
        var (float): fitted variance (variance at termination)
        eps (float, optional): small parameter to prevent division by zero. Defaults to 1e-12.

    Returns:
        Float[Array, "i d"]: vectors at interpolated points
    """
    nu = jnp.clip(jnp.sum(P, axis=1), eps)
    psi = (lambda_ * var / s**2) * jnp.diag(1.0 / nu)
    G = affinity_matrix(interp, mov, kernel, beta)
    return G @ jnp.linalg.inv(G_mm + psi) @ resid


def interpolate_covariance(
    mov: Float[Array, "m d"],
    interp: Float[Array, "i d"],
    P: Float[Array, "m n"],
    G_mm: Float[Array, "m m"],
    kernel: KernelFunction,
    beta: float,
    s: float,
    lambda_: float,
    var: float,
    eps: float = 1e-12,
) -> Float[Array, "i i"]:
    """Predict the covariance matrix between all interpolation points.

    Args:
        mov (Float[Array, "m d"]): points from moving point cloud
        interp (Float[Array, "i d"]): points to interpolate vectors at
        P (Float[Array, "m n"]): fitted matching matrix
        G_mm (Float[Array, "m m"]): gram matrix between all pairs of points in moving point cloud
        kernel (KernelFunction): kernel function
        beta (float): shape parameter for kernel
        s (float): fitted scaling term
        lambda_ (float): regularization parameter
        var (float): fitted variance (variance at termination)
        eps (float, optional): small parameter to prevent division by zero. Defaults to 1e-12.

    Returns:
        Float[Array, "i i"]: covariance matrix
    """
    nu = jnp.clip(jnp.sum(P, axis=1), eps)
    psi = (lambda_ * var / s**2) * jnp.diag(1.0 / nu)
    G_im = affinity_matrix(interp, mov, kernel, beta)
    G_ii = affinity_matrix(interp, interp, kernel, beta)
    return (1.0 / lambda_) * (
        G_ii - G_im @ jnp.linalg.inv(G_mm + psi) @ jnp.transpose(G_im)
    )


def invert_gp_bcpd_mapping(
    y: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    resid: Float[Array, "m d"],
    P: Float[Array, "m n"],
    G_mm: Float[Array, "m m"],
    kernel: KernelFunction,
    beta: float,
    s: float,
    lambda_: float,
    var: float,
    max_iter: int = 20,
    tol: float = 1e-6,
    eps: float = 1e-12,
) -> Float[Array, "n d"]:
    """Invert the BCPD GP mapping y = x + v(x) to find x.

    The forward map v is the BCPD GP posterior mean:

    .. code-block:: text

        v(x) = G(x, mov) @ inv(G_mm + Psi) @ resid

    where ``Psi = (lambda_ * var / s**2) * diag(1 / nu)`` and
    ``nu = sum(P, axis=1)`` (clipped to avoid division by zero).

    The inverse is found with Newton-Raphson iteration:

    .. code-block:: text

        x <- x - inv(I + J_v(x)) @ (x + v(x) - y)

    This is consistent with :func:`interpolate` by construction: both
    functions compute ``Psi`` from the same BCPD state variables, so the
    forward model used here exactly matches the one used during registration.

    Args:
        y (Float[Array, "n d"]): points in the target/deformed space to invert.
        mov (Float[Array, "m d"]): control points (moving cloud), kernel evaluation sites.
        resid (Float[Array, "m d"]): fitting residual at control points.
        P (Float[Array, "m n"]): matching matrix from the BCPD E-step (m_mov x n_ref).
        G_mm (Float[Array, "m m"]): Gram matrix between all pairs of control points.
        kernel (KernelFunction): kernel function.
        beta (float): kernel shape parameter.
        s (float): fitted isotropic scaling term.
        lambda_ (float): regularization parameter.
        var (float): fitted noise variance at convergence.
        max_iter (int, optional): maximum Newton iterations. Defaults to 20.
        tol (float, optional): convergence tolerance (RMS change in x). Defaults to 1e-6.
        eps (float, optional): small value to prevent division by zero in nu. Defaults to 1e-12.

    Returns:
        Float[Array, "n d"]: inverted source-space points.
    """
    x_final, _ = _invert_with_jacobian(
        y,
        mov,
        resid,
        P,
        G_mm,
        kernel,
        beta,
        s,
        lambda_,
        var,
        max_iter,
        tol,
        eps,
    )
    return x_final


def _compute_gp_weights(
    resid: Float[Array, "m d"],
    P: Float[Array, "m n"],
    G_mm: Float[Array, "m m"],
    s: float,
    lambda_: float,
    var: float,
    eps: float = 1e-12,
) -> Float[Array, "m d"]:
    """Compute GP weights W = inv(G + psi) @ resid.

    Args:
        resid: Fitting residual (m, d).
        P: Matching matrix (m, n).
        G_mm: Gram matrix (m, m).
        s: Fitted scaling term.
        lambda_: Regularization parameter.
        var: Converged variance.
        eps: Small value for numerical stability.

    Returns:
        Float[Array, "m d"]: GP weight matrix.
    """
    nu = jnp.clip(jnp.sum(P, axis=1), eps)  # (m,)
    psi = (lambda_ * var / s**2) * jnp.diag(1.0 / nu)  # (m, m)
    return jnp.linalg.solve(G_mm + psi, resid)  # (m, d)


def interpolate_variance(
    mov: Float[Array, "m d"],
    interp: Float[Array, "i d"],
    P: Float[Array, "m n"],
    G_mm: Float[Array, "m m"],
    kernel: KernelFunction,
    beta: float,
    s: float,
    lambda_: float,
    var: float,
    eps: float = 1e-12,
) -> Float[Array, " i"]:
    """Compute posterior variance of the interpolated displacement field.

    For each query point ``z``, computes the per-coordinate variance
    ``sigma_v^2(z) = (1/lambda) * (K(z,z) - k(z)^T inv(G + psi) k(z))``
    where ``psi = (lambda * var / s^2) * diag(1/nu)`` and
    ``nu = sum(P, axis=1)``.

    Args:
        mov: Moving point cloud from alignment (m, d).
        interp: Query points for interpolation (i, d).
        P: Matching matrix from alignment (m, n).
        G_mm: Gram matrix between moving points (m, m).
        kernel: Kernel function.
        beta: Kernel shape parameter.
        s: Fitted scaling term.
        lambda_: Regularization parameter.
        var: Converged variance from alignment.
        eps: Small value to prevent division by zero.

    Returns:
        Float[Array, " i"]: Per-point, per-coordinate posterior variance.

    Notes:
        The variance is isotropic across output dimensions. For a D-dimensional
        problem, the total variance at point z_i is D * sigma_v^2(z_i).
    """
    nu = jnp.clip(jnp.sum(P, axis=1), eps)  # (m,)
    psi = (lambda_ * var / s**2) * jnp.diag(1.0 / nu)  # (m, m)
    G_reg = G_mm + psi  # (m, m)
    G_im = affinity_matrix(interp, mov, kernel, beta)  # (i, m)
    # Solve G_reg @ alpha = G_im^T  =>  alpha has shape (m, i)
    alpha = jnp.linalg.solve(G_reg, G_im.T)  # (m, i)
    # K(z, z) for each query point
    K_zz = jax.vmap(lambda z_i: kernel(z_i[None, :], z_i[None, :], beta))(
        interp
    )  # (i,)
    # sigma_v^2(z_i) = (1/lambda) * (K(z_i, z_i) - k(z_i)^T alpha_i)
    variance = (1.0 / lambda_) * (
        K_zz - jnp.sum(G_im * alpha.T, axis=1)
    )  # (i,)
    return jnp.maximum(variance, 0.0)


def _jacobian_forward(
    interp: Float[Array, "i d"],
    mov: Float[Array, "m d"],
    resid: Float[Array, "m d"],
    G_mm: Float[Array, "m m"],
    P: Float[Array, "m n"],
    kernel: KernelFunction,
    beta: float,
    s: float,
    lambda_: float,
    var: float,
    eps: float = 1e-12,
) -> Float[Array, "i d d"]:
    """Compute the Jacobian of the forward transformation T(z) = z + v(z).

    J_T(z) = I_D + (dk(z)/dz)^T @ W_gp
    where W_gp = inv(G + psi) @ resid.

    For the Gaussian kernel:
    dK(z, y_m)/dz = -(1/beta) * K(z, y_m) * (z - y_m).

    Args:
        interp: Query points (i, d).
        mov: Moving (control) points (m, d).
        resid: Fitting residual (m, d).
        G_mm: Gram matrix between moving points (m, m).
        P: Matching matrix (m, n).
        kernel: Kernel function.
        beta: Kernel shape parameter.
        s: Fitted scaling term.
        lambda_: Regularization parameter.
        var: Converged variance.
        eps: Small value for numerical stability.

    Returns:
        Float[Array, "i d d"]: Jacobian matrices J_T(z_i) for each query point.
    """
    d = interp.shape[1]
    W_gp = _compute_gp_weights(resid, P, G_mm, s, lambda_, var, eps)  # (m, d)

    def _jacobian_single(z_i):
        """Compute J_T for a single query point."""
        diff = z_i[None, :] - mov  # (m, d)
        k = jax.vmap(lambda y_m: kernel(z_i[None, :], y_m[None, :], beta))(
            mov
        )  # (m,)
        grad_k = -(1.0 / beta) * k[:, None] * diff  # (m, d)
        return jnp.eye(d) + W_gp.T @ grad_k  # (d, d)

    return jax.vmap(_jacobian_single)(interp)  # (i, d, d)


def _invert_with_jacobian(
    y: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    resid: Float[Array, "m d"],
    P: Float[Array, "m n"],
    G_mm: Float[Array, "m m"],
    kernel: KernelFunction,
    beta: float,
    s: float,
    lambda_: float,
    var: float,
    max_iter: int = 20,
    tol: float = 1e-6,
    eps: float = 1e-12,
) -> tuple[Float[Array, "n d"], Float[Array, "n d d"]]:
    """Invert the GP mapping and return both inverted points and final Jacobians.

    Args:
        y: Points to invert (target space) (n, d).
        mov: Control points (m, d).
        resid: Fitting residual (m, d).
        P: Matching matrix (m, n).
        G_mm: Gram matrix (m, m).
        kernel: Kernel function.
        beta: Kernel shape parameter.
        s: Fitted scaling term.
        lambda_: Regularization parameter.
        var: Converged variance.
        max_iter: Maximum Newton iterations.
        tol: Convergence tolerance (RMS change in x).
        eps: Small value for numerical stability.

    Returns:
        tuple[Float[Array, "n d"], Float[Array, "n d d"]]:
            - inverted_points: Source-space points (n, d).
            - jacobians: Forward Jacobian J_T(x) at each inverted point (n, d, d).
    """
    W_gp = _compute_gp_weights(resid, P, G_mm, s, lambda_, var, eps)  # (m, d)

    def vector_field(x_point: Float[Array, " d"]) -> Float[Array, " d"]:
        g = jax.vmap(lambda m: kernel(x_point[None, :], m[None, :], beta))(mov)
        return g @ W_gp

    def h_func(
        x_point: Float[Array, " d"], y_target: Float[Array, " d"]
    ) -> Float[Array, " d"]:
        return x_point + vector_field(x_point) - y_target

    def newton_step(x_point: Float[Array, " d"], y_target: Float[Array, " d"]):
        h = h_func(x_point, y_target)
        # Use closed-form Jacobian
        J = _jacobian_forward(
            x_point[None, :],
            mov,
            resid,
            G_mm,
            P,
            kernel,
            beta,
            s,
            lambda_,
            var,
            eps,
        )[0]
        return x_point - jnp.linalg.solve(J, h), J

    vmap_newton_step = jax.vmap(newton_step)

    # Initial guess: x = y; diff seeded above tol to ensure at least one step
    init_val = (y, jnp.inf, 0, jnp.zeros((y.shape[0], y.shape[1], y.shape[1])))

    def cond_fun(
        state: tuple[
            Float[Array, "n d"], Float[Array, ""], int, Float[Array, "n d d"]
        ],
    ) -> Bool:
        _, diff, i, _ = state
        return jnp.logical_and(i < max_iter, diff > tol)

    def body_fun(
        state: tuple[
            Float[Array, "n d"], Float[Array, ""], int, Float[Array, "n d d"]
        ],
    ) -> tuple[
        Float[Array, "n d"], Float[Array, ""], int, Float[Array, "n d d"]
    ]:
        x_curr, _, i, _ = state
        x_next, J_final = vmap_newton_step(x_curr, y)
        diff = jnp.sqrt(jnp.mean(jnp.square(x_next - x_curr)))
        return x_next, diff, i + 1, J_final

    x_final, _, _, J_final = jax.lax.while_loop(
        cond_fun, body_fun, init_val  # pyright: ignore[reportArgumentType]
    )

    return x_final, J_final


def interpolate_variance_inverse(
    mov: Float[Array, "m d"],
    target: Float[Array, "n d"],
    P: Float[Array, "m n"],
    G_mm: Float[Array, "m m"],
    resid: Float[Array, "m d"],
    kernel: KernelFunction,
    beta: float,
    s: float,
    lambda_: float,
    var: float,
    eps: float = 1e-12,
    inv_max_iter: int = 20,
    inv_tol: float = 1e-6,
) -> tuple[Float[Array, "n d"], InverseCovariance]:
    """Compute covariance of the inverse transformation at target points.

    For each target point ``x``, finds the source point ``z*`` such that
    ``T(z*) = x``, then computes the covariance of ``z*`` using the
    delta method:

    ``Cov[z*] = sigma_v^2(z_hat) * J_T(z_hat)^{-1} * J_T(z_hat)^{-T}``

    where ``J_T`` is the Jacobian of the forward transformation and
    ``sigma_v^2`` is the forward displacement variance.

    Args:
        mov: Moving point cloud from alignment (m, d).
        target: Target points in deformed space (n, d).
        P: Matching matrix from alignment (m, n).
        G_mm: Gram matrix between moving points (m, m).
        resid: Fitting residual (m, d).
        kernel: Kernel function.
        beta: Kernel shape parameter.
        s: Fitted scaling term.
        lambda_: Regularization parameter.
        var: Converged variance from alignment.
        eps: Small value for numerical stability.
        inv_max_iter: Maximum Newton iterations for inversion.
        inv_tol: Convergence tolerance for inversion.

    Returns:
        tuple[Float[Array, "n d"], InverseCovariance]:
            - inverted_points: Source-space points z* for each target (n, d).
            - inverse_covariance: Covariance matrix for each inverted point (n, d, d).

    Notes:
        When the forward map is locally rigid (J_T approx I), the inverse
        covariance is approximately isotropic: ``Cov[z*] approx sigma_v^2 * I_D``.
        Under strong compression or shearing, the covariance becomes
        anisotropic, amplified along directions where the forward map
        contracts.
    """
    # Step 1: Invert the MAP field, getting both points and Jacobians
    z_hat, J_T = _invert_with_jacobian(
        target,
        mov,
        resid,
        P,
        G_mm,
        kernel,
        beta,
        s,
        lambda_,
        var,
        inv_max_iter,
        inv_tol,
        eps,
    )

    # Step 2: Compute forward variance at the inverted points
    sigma_v_sq = interpolate_variance(
        mov, z_hat, P, G_mm, kernel, beta, s, lambda_, var, eps
    )  # (n,)

    # Step 3: Form inverse covariance: Cov = sigma_v^2 * J^{-1} * J^{-T}
    d = target.shape[1]
    I_d = jnp.eye(d)

    def solve_jacobian(J_single: Float[Array, "d d"]) -> Float[Array, "d d"]:
        M = jnp.linalg.solve(J_single, I_d)
        return M @ M.T

    vmap_solve = jax.vmap(solve_jacobian)
    MM_T = vmap_solve(J_T)  # (n, d, d)

    inverse_cov = sigma_v_sq[:, None, None] * MM_T  # (n, d, d)
    return z_hat, inverse_cov
