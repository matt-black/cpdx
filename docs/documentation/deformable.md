# Deformable (non-linear) Registration

Fit a (possibly non-linear) vector field that maps the moving point set onto the reference set of points. This is done through a regularized, Gaussian-process like transform. The transformation field is smoothed and forced to maintain the relative position of points through the transform by motion coherence ([1,2]).

For details, see Section 5 and Figure 4 of [2].

The `TransformParams` object returned by `align` and `align_fixed_iter` is a `tuple` of three arrays:

1. `MatchingMatrix` of shape `(m,n)` where the `ij`-th value of the array is the probability that point `n` in the moving set is matched with point `m` from the reference set.
2. `AffineMatrix` of shape `(d,d)` where `d` is the dimension of the point sets (usually 2 or 3).
3. `Translation`, a `d`-length vector representing the translation. Note that this translation is at the scale of the reference point set (in the forward transform, it is applied *after* rotation/shearing/scaling).

::: cpdx.deformable.align
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.deformable.align_fixed_iter
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.deformable.transform
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.deformable.interpolate
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.deformable.maximization
    handler: python
    options:
        show_source: false
        show_root_heading: true

## References

[1] Yuille, Alan L., and Norberto M. Grzywacz. "A mathematical analysis of the motion coherence theory." International Journal of Computer Vision 3.2 (1989): 155-175.

[2] Myronenko, Andriy, and Xubo Song. "Point set registration: Coherent point drift." IEEE transactions on pattern analysis and machine intelligence 32.12 (2010): 2262-2275.
