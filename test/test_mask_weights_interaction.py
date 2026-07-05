"""Diagnostic test: verify how the align functions handle weights + mask interaction."""

import jax.numpy as jnp

from cpdx._matching import expectation, expectation_masked, expectation_weighted


def test_mask_weights_pattern():
    """Check current handling pattern in align functions.

    The rigid.py / affine.py / nonrigid.py all use the same pattern:
      if moving_weights is None:
          if mask is None:
              P = expectation(...)
          else:
              P = expectation_masked(...)
      else:
          P = expectation_weighted(...)
    
    This means: when weights are provided, mask is IGNORED.
    """
    x = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    y_t = jnp.array([[0.1, 0.1], [1.1, 1.1]])
    var = jnp.array(0.1)
    w = 0.01

    # Test 1: bare expectation
    P_bare = expectation(x, y_t, var, w)
    print("Bare expectation:\n", P_bare)

    # Test 2: masked expectation (diagonal mask)
    mask = jnp.eye(2)
    P_masked = expectation_masked(x, y_t, var, w, mask)
    print("Masked expectation (diagonal mask):\n", P_masked)

    # Test 3: weighted expectation (uniform weights)
    weights = jnp.array([1.0, 1.0])
    P_weighted = expectation_weighted(x, y_t, var, w, weights)
    print("Weighted expectation (uniform weights):\n", P_weighted)

    # Verify that P_weighted with uniform weights matches bare expectation
    # (uniform weights should be equivalent to no weights)
    assert jnp.allclose(P_bare, P_weighted, atol=1e-6), \
        "Uniform weights should match bare expectation"

    # Verify that mask zeroes out off-diagonal
    assert P_masked[0, 1] == 0 and P_masked[1, 0] == 0, \
        "Masked entries should be 0"

    print("\nPATTERN CONFIRMED:")
    print("  expectation()   - no weights, no mask")
    print("  expectation_masked()  - no weights, with mask")
    print("  expectation_weighted() - with weights, NO mask support")
    print("")
    print("The current code uses an if-else pattern:")
    print("  if moving_weights is None:")
    print("      if mask is None: P = expectation(...)")
    print("      else:            P = expectation_masked(...)")
    print("  else:")
    print("      P = expectation_weighted(...)  # mask is IGNORED")


def test_combined_expectation_gap():
    """Demonstrate the missing case: weights + mask + uncertainty.

    The paper's Section 7 describes a single general formula that handles
    all three simultaneously. The current library has 3 separate functions
    that can't be composed.
    """
    x = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    y_t = jnp.array([[0.1, 0.1], [1.1, 1.1]])
    var = jnp.array(0.1)
    w = 0.01
    mask = jnp.array([[1, 0], [1, 1]], dtype=float)  # mov[0] can only match ref[0]
    weights = jnp.array([2.0, 1.0])  # mov[0] is 2x more important

    # What the user might want: apply BOTH weights AND mask
    # Current library: impossible without modifying code

    # Workaround: bake mask into weights? No, that doesn't work because
    # expectation_weighted uses alpha_m[m] * exp(...) but doesn't support masking
    # individual pairs.

    print("Combined weights + mask: NOT SUPPORTED in current library.")
    print("The expectation_weighted function does not accept a mask parameter.")


if __name__ == "__main__":
    test_mask_weights_pattern()
    print()
    test_combined_expectation_gap()
