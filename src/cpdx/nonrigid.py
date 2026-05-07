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
    "KernelMatrix",
    "CoeffMatrix",
    "TransformParams",
    "PointVariance",
    "InverseCovariance",
    "align",
    "align_fixed_iter",
    "transform",
    "interpolate",
    "interpolate_variance",
    "interpolate_variance_inverse",
    "maximization",
    "invert_gp_mapping",
]


type KernelMatrix = Float[Array, "m m"]
type CoeffMatrix = Float[Array, "m d"]
type TransformParams = tuple[MatchingMatrix, KernelMatrix, CoeffMatrix]
type PointVariance = Float[Array, " n"]
type InverseCovariance = Float[Array, "n d d"]


def align(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    outlier_prob: float,
    regularization_param: float,
    kernel_var: float,
    max_iter: int,
    tolerance: float,
    moving_weights: Float[Array, " m"] | None = None,
    mask: Float[Array, "m n"] | None = None,
) -> tuple[TransformParams, tuple[Float[Array, ""], int]]:
    """Align the moving points onto the reference points by a nonrigid transform.

    Args:
        ref (Float[Array, "n d"]): reference points
        mov (Float[Array, "m d"]): moving points
        outlier_prob (float): outlier probability, should be in range [0,1].
        regularization_param (float): regularization parameter (usually termed "lambda" in the literature) for motion coherence.
        kernel_var (float): variance of Gaussian kernel function.
        max_iter (int): maximum # of iterations to optimize for.
        tolerance (float): tolerance for matching variance, below which the algorithm will terminate.
        moving_weights (Float[Array, " m"] | None): optional per-point weights for source points (arbitrary positive values). If None, uniform weights are used.
        mask (Float[Array, "m n"] | None): optional mask matrix where nonzero entries indicate valid matches.

    Returns:
        tuple[TransformParams, tuple[Float[Array, ""], int]]: the fitted transform parameters (the matching matrix and the kernel and coefficient matrices) along with the final variance and the number of iterations that the algorithm was run for.
    """
    # initialize variance
    n, d = ref.shape
    m, _ = mov.shape
    var_i = jnp.sum(sqdist(ref, mov)) / (m * n * d)

    # compute gaussian kernel matrix
    G = jnp.exp(jnp.negative(jnp.divide(sqdist(mov, mov), 2 * kernel_var)))

    def cond_fund(
        a: tuple[
            tuple[Float[Array, "m d"], Float[Array, "m m"]],
            tuple[Float[Array, ""], int],
        ],
    ) -> Bool:
        _, (var, iter_num) = a
        return jnp.logical_and(var > tolerance, iter_num < max_iter)

    def body_fund(
        a: tuple[
            tuple[CoeffMatrix, MatchingMatrix], tuple[Float[Array, ""], int]
        ],
    ) -> tuple[
        tuple[CoeffMatrix, MatchingMatrix], tuple[Float[Array, ""], int]
    ]:
        (W, _), (var, iter_num) = a
        mov_t = transform(mov, G, W)
        if moving_weights is None:
            if mask is None:
                P = expectation(ref, mov_t, var, outlier_prob)
            else:
                P = expectation_masked(ref, mov_t, var, outlier_prob, mask)
        else:
            P = expectation_weighted(
                ref, mov_t, var, outlier_prob, moving_weights
            )
        W, new_var = maximization(
            ref, mov, P, G, var, regularization_param, tolerance
        )
        return ((W, P), (new_var, iter_num + 1))

    (W, P), (var_f, num_iter) = jax.lax.while_loop(
        cond_fund,
        body_fund,
        ((jnp.zeros_like(mov), jnp.zeros((m, n))), (var_i, 0)),
    )
    return (P, G, W), (var_f, num_iter)


def align_fixed_iter(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    outlier_prob: float,
    regularization_param: float,
    kernel_var: float,
    num_iter: int,
    moving_weights: Float[Array, " m"] | None = None,
    mask: Float[Array, "m n"] | None = None,
) -> tuple[TransformParams, Float[Array, " {num_iter}"]]:
    """Align the moving points onto the reference points by a nonrigid transform.

    Args:
        ref (Float[Array, "n d"]): reference points
        mov (Float[Array, "m d"]): moving points
        outlier_prob (float): outlier probability, should be in range [0,1].
        regularization_param (float): regularization parameter (usually termed "lambda" in the literature) for motion coherence.
        kernel_var (float): variance of Gaussian kernel function.
        num_iter (int): # of iterations to optimize for.
        moving_weights (Float[Array, " m"] | None): optional per-point weights for source points (arbitrary positive values). If None, uniform weights are used.
        mask (Float[Array, "m n"] | None): optional mask matrix where nonzero entries indicate valid matches.

    Returns:
        tuple[TransformParams, Float[Array, " {num_iter}"]]: the fitted transform parameters (the matching matrix and the kernel and coefficient matrices) along with the variance at each step of the optimization.
    """
    n, d = ref.shape
    m, _ = mov.shape
    # compute gaussian kernel
    G = jnp.exp(jnp.negative(jnp.divide(sqdist(mov, mov), 2 * kernel_var)))
    var_i = (jnp.sum(sqdist(ref, mov)) / (m * n * d)).item()

    def scan_fun(
        a: tuple[tuple[MatchingMatrix, CoeffMatrix], Float[Array, ""]],
        _,
    ):
        (_, W), var = a
        mov_t = transform(mov, G, W)
        if moving_weights is None:
            if mask is None:
                P = expectation(ref, mov_t, var, outlier_prob)
            else:
                P = expectation_masked(ref, mov_t, var, outlier_prob, mask)
        else:
            P = expectation_weighted(
                ref, mov_t, var, outlier_prob, moving_weights
            )
        W, new_var = maximization(
            ref, mov, P, G, var, regularization_param, 1e-6
        )
        return ((P, W), new_var), new_var

    ((P, W), _), varz = jax.lax.scan(
        scan_fun,
        ((jnp.zeros((m, n)), jnp.zeros_like(mov)), var_i),
        length=num_iter,
    )

    return (P, G, W), varz


def maximization(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    P: MatchingMatrix,
    G: KernelMatrix,
    var: Float[Array, ""],
    regularization_param: float,
    tolerance: float,
) -> tuple[CoeffMatrix, Float[Array, ""]]:
    """Do a single M-step.

    Args:
        x (Float[Array, "n d"]): target point set
        y (Float[Array, "m d"]): source point set
        P (MatchingMatrix): matching matrix
        G (KernelMatrix): matrix of kernel values between points in the source point set
        var (Float[Array, ""]): current variance
        regularization_param (float): regularization parameter (usually termed "lambda" in the literature) for motion coherence.
        tolerance (float): termination tolerance

    Returns:
        tuple[tuple[AffineMatrix, Translation], Float[Array, ""]]:
    """

    W = update_transform(x, y, P, G, var, regularization_param)
    y_t = transform(y, G, W)
    new_var = update_variance(x, y_t, P, tolerance)
    return W, new_var


def transform(
    y: Float[Array, "m d"],
    G: KernelMatrix,
    W: CoeffMatrix,
) -> Float[Array, "m d"]:
    """Transform the input points by nonrigid warping.

    Args:
        y (Float[Array, "m d"]): `d`-dimensional points to be transformed
        G (KernelMatrix): matrix of kernel values between points (should be m x m)
        W (CoeffMatrix): fitted coefficient matrix (should be same shape as `y`)

    Returns:
        Float[Array, "m d"]: warped points, `y + G @ W`
    """
    return jnp.add(y, G @ W)


def update_transform(
    x: Float[Array, "n d"],
    y: Float[Array, "m d"],
    P: MatchingMatrix,
    G: KernelMatrix,
    var: Float[Array, ""],
    regularization_param: float,
) -> CoeffMatrix:
    m, _ = P.shape
    P1 = jnp.sum(P, axis=1)
    A = jnp.diag(P1) @ G + regularization_param * var * jnp.eye(m)
    B = jnp.matmul(P, x) - jnp.diag(P1) @ y
    return jnp.linalg.solve(A, B)


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


def interpolate(
    mov: Float[Array, "m d"],
    interp: Float[Array, "n d"],
    W: Float[Array, "m d"],
    kernel_var: float,
) -> Float[Array, "n d"]:
    """Interpolate values of vector field at points outside of the original fit domain.

    Args:
        mov (Float[Array, "m d"]): "moving" point cloud that was aligned
        interp (Float[Array, "n d"]): points to interpolate
        W (Float[Array, "m d"]): fitted transform coefficients
        kernel_var (float): variance of kernel

    Returns:
        Float[Array, "n d"]: interpolated vector values at specified points

    Notes:
        This assumes a "Gaussian process-like" interpretation of the fitting coefficients whereby the deformation field can be interpolated by computing a Gram matrix between the interpolated points and the moving points used during fitting, then the fitted weights are used to calculate deformation vectors at the interpolation coordinates.
    """
    # calculate kernel matrix b/t moving & interpolating points
    G_im = jnp.exp(
        jnp.negative(jnp.divide(sqdist(interp, mov), 2 * kernel_var))
    )
    return G_im @ W


def interpolate_variance(
    mov: Float[Array, "m d"],
    interp: Float[Array, "n d"],
    P: MatchingMatrix,
    G: KernelMatrix,
    kernel_var: float,
    regularization_param: float,
    var: float,
    method: str = "cholesky",
    rank: int | None = None,
    eps: float = 1e-3,
) -> PointVariance:
    """Compute posterior variance of the interpolated displacement field.

    Under the empirical Bayes approximation, the posterior precision matrix for
    the coefficient weights is ``H = (1/sigma^2) * G * D * G + lambda * G``,
    where ``D = diag(sum(P, axis=1))``. The per-coordinate variance at a query
    point ``z`` is ``sigma_v^2(z) = k(z)^T H^{-1} k(z)``.

    Args:
        mov: Moving point cloud from alignment (m, d).
        interp: Query points for interpolation (n, d).
        P: Posterior probability matrix from alignment (m, n_ref).
        G: Gram matrix between moving points (m, m).
        kernel_var: Gaussian kernel width parameter beta^2.
        regularization_param: Regularization parameter lambda.
        var: Converged variance sigma^2 from alignment.
        method: Numerical method -- "cholesky" or "low_rank".
        rank: Rank K for low-rank approximation (default: auto-select as min(100, m)).
        eps: Small nugget added to H for numerical stability.

    Returns:
        PointVariance: Per-point, per-coordinate posterior variance sigma_v^2(z) of shape (n,).

    Notes:
        The variance is isotropic across output dimensions. For a D-dimensional
        problem, the total variance at point z_i is D * sigma_v^2(z_i).
    """
    m = mov.shape[0]
    nu = jnp.sum(P, axis=1)  # (m,) effective matching counts

    # Cross-kernel matrix between interpolation and moving points
    G_im = jnp.exp(
        jnp.negative(jnp.divide(sqdist(interp, mov), 2 * kernel_var))
    )  # (n, m)

    if method == "cholesky":
        return _interpolate_variance_cholesky(
            G, nu, G_im, var, regularization_param, eps
        )
    elif method == "low_rank":
        K = rank or min(100, m)
        return _interpolate_variance_low_rank(
            G, nu, G_im, var, regularization_param, K, eps
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'cholesky' or 'low_rank'."
        )


def _interpolate_variance_cholesky(
    G: KernelMatrix,
    nu: Float[Array, " m"],
    G_im: Float[Array, "n m"],
    var: float,
    regularization_param: float,
    eps: float,
) -> PointVariance:
    """Cholesky-based computation of interpolation variance.

    Forms H = (1/var) * G * diag(nu) * G + lambda * G + eps * I,
    computes Cholesky factor L, then solves L @ Y^T = G_im^T.
    Variance is row-wise squared norm of Y.
    """
    m = G.shape[0]
    # H = (1/var) * G @ diag(nu) @ G + lambda * G + eps * I
    # G @ diag(nu) scales rows of the result: (G * nu[:, None]) @ G
    # Equivalently: G @ (diag(nu) @ G) where diag(nu) @ G scales rows of G
    GDG = G @ (nu[:, None] * G)
    H = (1.0 / var) * GDG + regularization_param * G
    H = H + eps * jnp.eye(m)
    L = jnp.linalg.cholesky(H)
    # Solve L @ Y^T = G_im^T  =>  Y^T = L^{-1} @ G_im^T
    Y_T = jnp.linalg.solve(L, G_im.T)  # (m, n)
    # sigma_v^2(z_i) = ||y_i||^2 where y_i is the i-th column of Y^T
    variance = jnp.sum(Y_T * Y_T, axis=0)  # (n,)
    return jnp.maximum(variance, 0.0)


def _interpolate_variance_low_rank(
    G: KernelMatrix,
    nu: Float[Array, " m"],
    G_im: Float[Array, "n m"],
    var: float,
    regularization_param: float,
    K: int,
    eps: float = 1e-3,
) -> PointVariance:
    """Low-rank computation of interpolation variance.

    Uses truncated eigendecomposition G ~ Q * Lambda * Q^T with K << M,
    projects H into the low-rank basis, and solves in K-dimensional space.

    H_K = (1/sigma^2) * Lambda * M_K * Lambda + lambda * Lambda + eps * I
    where M_K = Q^T * diag(nu) * Q
    Then sigma_v^2(z) = k_tilde^T * H_K^{-1} * k_tilde
    where k_tilde = G_im @ Q
    """
    m = G.shape[0]
    K = min(K, m)

    # Truncated eigendecomposition
    eigenvalues, eigenvectors = jnp.linalg.eigh(G)  # (m,), (m, m)
    # Take K largest eigenvalues
    idx = jnp.argsort(eigenvalues)[::-1][:K]
    eigvals_K = eigenvalues[idx]  # (K,)
    Q = eigenvectors[:, idx]  # (m, K)

    # M_K = Q^T * diag(nu) * Q  =>  (K, K)
    M_K = Q.T @ (nu[:, None] * Q)

    # H_K = (1/var) * Lambda * M_K * Lambda + lambda * Lambda + eps * I  =>  (K, K)
    Lambda = jnp.diag(eigvals_K)
    H_K = (
        (1.0 / var) * Lambda @ M_K @ Lambda
        + regularization_param * Lambda
        + eps * jnp.eye(K)
    )

    # k_tilde = G_im @ Q  =>  (n, K)
    k_tilde = G_im @ Q

    # Solve H_K @ alpha = k_tilde^T for each query point
    alpha = jnp.linalg.solve(H_K, k_tilde.T)  # (K, n)

    # sigma_v^2(z_i) = k_tilde[i] @ alpha[:, i]
    variance = jnp.sum(k_tilde * alpha.T, axis=1)  # (n,)
    return jnp.maximum(variance, 0.0)


def _jacobian_forward(
    z: Float[Array, " n d"],
    mov: Float[Array, "m d"],
    W: CoeffMatrix,
    kernel_var: float,
) -> Float[Array, "n d d"]:
    """Compute the Jacobian of the forward transformation T(z) = z + v(z).

    J_T(z) = I_D + sum_m w_m * grad_k_m(z)^T
    where grad_k_m(z) = -(1/beta^2) * G(z, y_m) * (z - y_m).

    Args:
        z: Query points (n, d).
        mov: Moving (control) points (m, d).
        W: Fitted coefficient matrix (m, d).
        kernel_var: Gaussian kernel width parameter beta^2.

    Returns:
        Float[Array, "n d d"]: Jacobian matrices J_T(z_i) for each query point.
    """
    d = z.shape[1]

    def _jacobian_single(z_i):
        """Compute J_T for a single query point."""
        diff = z_i[None, :] - mov  # (m, d)
        k = jnp.exp(-jnp.sum(diff**2, axis=1) / (2 * kernel_var))  # (m,)
        grad_k = -(1.0 / kernel_var) * k[:, None] * diff  # (m, d)
        return jnp.eye(d) + W.T @ grad_k  # (d, d)

    return jax.vmap(_jacobian_single)(z)  # (n, d, d)


def _invert_with_jacobian(
    y: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    W: CoeffMatrix,
    kernel_var: float,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> tuple[Float[Array, "n d"], Float[Array, "n d d"]]:
    """Invert the GP mapping and return both inverted points and final Jacobians.

    Args:
        y: Points to invert (target/deformed space) (n, d).
        mov: Control points (moving point cloud) (m, d).
        W: Fitted GP weight matrix (m, d).
        kernel_var: Variance of the Gaussian kernel.
        max_iter: Maximum number of Newton iterations.
        tol: Convergence tolerance (RMS change in x).

    Returns:
        tuple[Float[Array, "n d"], Float[Array, "n d d"]]:
            - inverted_points: Source-space points x (n, d).
            - jacobians: Forward Jacobian J_T(x) evaluated at each inverted point (n, d, d).
    """

    def vector_field(x_point: Float[Array, " d"]) -> Float[Array, " d"]:
        g = jax.vmap(
            lambda m: jnp.exp(
                jnp.negative(
                    jnp.divide(
                        jnp.sum(jnp.square(jnp.subtract(x_point, m))),
                        2 * kernel_var,
                    )
                )
            )
        )(mov)
        return g @ W

    def h_func(
        x_point: Float[Array, " d"], y_target: Float[Array, " d"]
    ) -> Float[Array, " d"]:
        return x_point + vector_field(x_point) - y_target

    def newton_step(x_point: Float[Array, " d"], y_target: Float[Array, " d"]):
        h = h_func(x_point, y_target)
        # Use closed-form Jacobian: J_T = I + W^T @ grad_k
        J = _jacobian_forward(x_point[None, :], mov, W, kernel_var)[0]
        delta = jnp.linalg.solve(J, h)
        return x_point - delta, J

    vmap_newton_step = jax.vmap(newton_step)

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


def invert_gp_mapping(
    y: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    W: Float[Array, "m d"],
    kernel_var: float,
    max_iter: int = 10,
    tol: float = 1e-6,
) -> Float[Array, "n d"]:
    """Invert the GP mapping y = x + v(x) to find x using Newton-Raphson.

    The forward map is ``T(z) = z + G(z, mov) @ W`` where the kernel is the
    same Gaussian used throughout this module:
    ``K(a, b) = exp(-‖a - b‖² / (2 * kernel_var))``.

    Args:
        y (Float[Array, "n d"]): points to invert (target/deformed space).
        mov (Float[Array, "m d"]): control points (moving point cloud used during fitting).
        W (Float[Array, "m d"]): fitted GP weight matrix (from :func:`maximization` / :func:`align`).
        kernel_var (float): variance of the Gaussian kernel — must match the value used during registration.
        max_iter (int, optional): maximum number of Newton iterations. Defaults to 10.
        tol (float, optional): convergence tolerance (RMS change in x). Defaults to 1e-6.

    Returns:
        Float[Array, "n d"]: inverted points x (source space).
    """
    x_final, _ = _invert_with_jacobian(y, mov, W, kernel_var, max_iter, tol)
    return x_final


def interpolate_variance_inverse(
    mov: Float[Array, "m d"],
    target: Float[Array, "n d"],
    P: MatchingMatrix,
    G: KernelMatrix,
    W: CoeffMatrix,
    kernel_var: float,
    regularization_param: float,
    var: float,
    method: str = "cholesky",
    rank: int | None = None,
    eps: float = 1e-3,
    inv_max_iter: int = 10,
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
        P: Posterior probability matrix from alignment (m, n_ref).
        G: Gram matrix between moving points (m, m).
        W: Fitted coefficient matrix (m, d).
        kernel_var: Gaussian kernel width parameter beta^2.
        regularization_param: Regularization parameter lambda.
        var: Converged variance sigma^2 from alignment.
        method: Numerical method for variance -- "cholesky" or "low_rank".
        rank: Rank K for low-rank approximation (default: auto-select).
        eps: Small nugget added to H for numerical stability.
        inv_max_iter: Maximum Newton iterations for inversion.
        inv_tol: Convergence tolerance for inversion.

    Returns:
        tuple[Float[Array, "n d"], InverseCovariance]:
            - inverted_points: Source-space points z* for each target (n, d).
            - inverse_covariance: Covariance matrix for each inverted point (n, d, d).

    Notes:
        When the forward map is locally rigid (J_T ≈ I), the inverse
        covariance is approximately isotropic: ``Cov[z*] ≈ sigma_v^2 * I_D``.
        Under strong compression or shearing, the covariance becomes
        anisotropic, amplified along directions where the forward map
        contracts.
    """
    # Step 1: Invert the MAP field, getting both points and Jacobians
    z_hat, J_T = _invert_with_jacobian(
        target, mov, W, kernel_var, inv_max_iter, inv_tol
    )

    # Step 2: Compute forward variance at the inverted points
    sigma_v_sq = interpolate_variance(
        mov,
        z_hat,
        P,
        G,
        kernel_var,
        regularization_param,
        var,
        method,
        rank,
        eps,
    )  # (n,)

    # Step 3: Form inverse covariance: Cov = sigma_v^2 * J^{-1} * J^{-T}
    # Solve J_T @ M = I for M = J_T^{-1}, then Cov = sigma_v^2 * M @ M^T
    d = target.shape[1]
    I_d = jnp.eye(d)

    def solve_jacobian(J_single: Float[Array, "d d"]) -> Float[Array, "d d"]:
        M = jnp.linalg.solve(J_single, I_d)
        return M @ M.T

    vmap_solve = jax.vmap(solve_jacobian)
    MM_T = vmap_solve(J_T)  # (n, d, d)

    inverse_cov = sigma_v_sq[:, None, None] * MM_T  # (n, d, d)
    return z_hat, inverse_cov
