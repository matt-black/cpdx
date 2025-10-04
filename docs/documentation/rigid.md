# Rigid registration

Fit a rigid transform that maps the moving point set onto the reference points.

Rigid transforms are linear transforms containing an isotropic scaling, rotation, and translation (`y=sRx + t`).

For details, see Section 4, Figure 2 of [1].

The `TransformParams` object returned by `align` and `align_fixed_iter` is a `tuple` of four arrays:

1. `MatchingMatrix` of shape `(m,n)` where the `ij`-th value of the array is the probability that point `n` in the moving set is matched with point `m` from the reference set.
2. `RotationMatrix` of shape `(d,d)` where `d` is the dimension of the point sets (usually 2 or 3).
3. `ScalingTerm`, a scalar array corresponding with the isotropic scaling term of the transform.
4. `Translation`, a `d`-length vector representing the translation. Note that this translation is at the scale of the reference point set (in the forward transform, it is applied *after* scaling).

::: cpdx.rigid.align
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.rigid.align_fixed_iter
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.rigid.transform
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.rigid.maximization
    handler: python
    options:
        show_source: false
        show_root_heading: true

## References

[1] Myronenko, Andriy, and Xubo Song. "Point set registration: Coherent point drift." IEEE transactions on pattern analysis and machine intelligence 32.12 (2010): 2262-2275.
