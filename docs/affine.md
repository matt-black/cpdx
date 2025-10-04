# Affine Registration

Fit an affine transform that maps the moving point set onto the reference set of points.
Affine transforms are linear transforms that allow rotation, translation, shear, and scaling in all axes.

For details, see Section 4 and Figure 3 of [1].

The `TransformParams` object returned by `align` and `align_fixed_iter` is a `tuple` of three arrays:

1. `MatchingMatrix` of shape `(m,n)` where the `ij`-th value of the array is the probability that point `n` in the moving set is matched with point `m` from the reference set.
2. `AffineMatrix` of shape `(d,d)` where `d` is the dimension of the point sets (usually 2 or 3).
3. `Translation`, a `d`-length vector representing the translation. Note that this translation is at the scale of the reference point set (in the forward transform, it is applied *after* rotation/shearing/scaling).

::: cpdx.affine.align
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.affine.align_fixed_iter
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.affine.transform
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.affine.maximization
    handler: python
    options:
        show_source: false
        show_root_heading: true

## References

[1] Myronenko, Andriy, and Xubo Song. "Point set registration: Coherent point drift." IEEE transactions on pattern analysis and machine intelligence 32.12 (2010): 2262-2275.
