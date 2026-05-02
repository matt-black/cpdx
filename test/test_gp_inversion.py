import jax
import jax.numpy as jnp

from cpdx.nonrigid import invert_gp_mapping


def test_gp_inversion_simple():
    # Setup control points
    mov = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    # Setup coefficients (some displacement)
    W = jnp.array([[0.1, 0.1], [-0.1, 0.0], [0.0, -0.1], [0.05, 0.05]])

    kernel_stddev = 1.0

    # Points in source space
    x_true = jnp.array([[0.5, 0.5], [0.2, 0.8], [0.7, 0.3]])

    # Forward mapping: T(x) = x + G(x, mov) @ W
    # using the same Gaussian kernel as registration: exp(-||a-b||^2 / (2 * kernel_stddev^2))
    def forward(x_pts):
        G = jax.vmap(
            lambda x: jax.vmap(
                lambda y: jnp.exp(
                    -jnp.sum(jnp.square(x - y)) / (2 * kernel_stddev**2)
                )
            )(mov)
        )(x_pts)
        return x_pts + G @ W

    y_target = forward(x_true)

    # Invert mapping
    x_recovered = invert_gp_mapping(
        y_target, mov, W, kernel_stddev, max_iter=20, tol=1e-7
    )

    print("\nTrue x:\n", x_true)
    print("Recovered x:\n", x_recovered)

    # Verify
    mse = jnp.mean(jnp.square(x_true - x_recovered))
    print(f"MSE: {mse}")

    assert mse < 1e-6


if __name__ == "__main__":
    test_gp_inversion_simple()
