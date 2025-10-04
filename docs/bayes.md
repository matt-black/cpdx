# Bayesian Coherent Point Drift (BCPD)

A fully probabilistic algorithm for coherent point drift. The algorithm simultaneously fits both a rigid and nonrigid ("deformable") transform. The full transform model is `y = sR(x + v) + t`.

See Fig. 2 of [1] for algorithmic details.

The `TransformParams` object returned by `align` and `align_with_ic` is a `tuple` of five arrays:

1. `MatchingMatrix` of shape `(m,n)` where the `ij`-th value of the array is the probability that point `n` in the moving set is matched with point `m` from the reference set.
2. `RotationMatrix` of shape `(d,d)` where `d` is the dimension of the point sets (usually 2 or 3).
3. `ScalingTerm`, a scalar array corresponding with the isotropic scaling term of the transform.
4. `Translation`, a `d`-length vector representing the translation. Note that this translation is at the scale of the reference point set (in the forward transform, it is applied *after* scaling).
5. `VectorField` vectors at each point in the moving point cloud that tell how to move that point during alignment. Note that these vectors are in the

::: cpdx.bayes.align
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.bayes.align_with_ic
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.bayes.transform
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.bayes.interpolate
    handler: python
    options:
        show_source: false
        show_root_heading: true

## Kernels

BCPD generalizes CPD by allowing non-Gaussian kernels. These functions all have the same arguments, described by the `KernelFunction` type which is `Callable[[Float[Array, "1 d"], Float[Array, "1 d"], float]]` where the first two parameters are single `d`-dimensional points and the last is a shape parameer for the kernel. Note that in some parts of the literature, these functions are also called "radial basis functions."

::: cpdx.bayes.kernel.gaussian_kernel
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.bayes.kernel.inverse_multiquadratic_kernel
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.bayes.kernel.rational_quadratic_kernel
    handler: python
    options:
        show_source: false
        show_root_heading: true

## Utilities

Utility functions for working with BCPD.

::: cpdx.bayes.util.affinity_matrix
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.bayes.util.initialize
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.bayes.util.apply_Tinv
    handler: python
    options:
        show_source: false
        show_root_heading: true

## References

[1] Hirose, Osamu. "A Bayesian formulation of coherent point drift." IEEE transactions on pattern analysis and machine intelligence 43.7 (2020): 2269-2286.

[2] Hirose, Osamu. "Acceleration of non-rigid point set registration with downsampling and Gaussian process regression." IEEE Transactions on Pattern Analysis and Machine Intelligence 43.8 (2020): 2858-2865.
