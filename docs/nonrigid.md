# Nonrigid (non-linear) Registration

Fit a (possibly non-linear) vector field that maps the moving point set onto the reference set of points. This is done through a regularized, Gaussian-process like transform. The transformation field is smoothed and forced to maintain the relative position of points through the transform by motion coherence ([1,2]).

For details, see Section 5 and Figure 4 of [2].

The `TransformParams` object returned by `align` and `align_fixed_iter` is a `tuple` of three arrays:

1. `MatchingMatrix` of shape `(m,n)` where the `ij`-th value of the array is the probability that point `n` in the moving set is matched with point `m` from the reference set.
2. `KernelMatrix` of shape `(m,m)` containing the Gaussian kernel values between all pairs of points in the moving point set.
3. `CoeffMatrix` of shape `(m,d)` containing the fitted coefficients for the nonrigid transform.

In addition to `TransformParams`, `align` returns a tuple of the final variance and the number of iterations performed. `align_fixed_iter` instead returns an array of the variance at each step of the optimization.

::: cpdx.nonrigid.align
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.nonrigid.align_fixed_iter
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.nonrigid.transform
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.nonrigid.interpolate
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.nonrigid.interpolate_variance
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.nonrigid.interpolate_variance_inverse
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.nonrigid.invert_gp_mapping
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.nonrigid.maximization
    handler: python
    options:
        show_source: false
        show_root_heading: true

## References

[1] Yuille, Alan L., and Norberto M. Grzywacz. "A mathematical analysis of the motion coherence theory." International Journal of Computer Vision 3.2 (1989): 155-175.

[2] Myronenko, Andriy, and Xubo Song. "Point set registration: Coherent point drift." IEEE transactions on pattern analysis and machine intelligence 32.12 (2010): 2262-2275.
