# RNR (Rigid + Non-Rigid) registration

Jointly fit a rigid transform and smooth non-rigid deformation that maps the moving point set onto the reference points.

The RNR model combines a global similarity transform (isotropic scaling, rotation, and translation: `y = sRx + t`) with a smooth local displacement field `v(x)` parameterized by a Gaussian process prior:

```
T(x) = s * R * x + t + v(x)
```

This joint estimation allows the soft correspondences to see the full transformation at every EM iteration, avoiding the sub-optimality of a sequential rigid-then-nonrigid pipeline. The GP prior naturally penalizes large-scale affine content in the displacement field, leaving the global rotation, scaling, and translation to the parametric part.

The `TransformParams` object returned by `align` and `align_fixed_iter` is a `tuple` of six arrays:

1. `MatchingMatrix` of shape `(m,n)` where the `ij`-th value of the array is the probability that point `n` in the reference set is matched with point `m` from the moving set.
2. `RotationMatrix` of shape `(d,d)` where `d` is the dimension of the point sets.
3. `ScalingTerm`, a scalar array corresponding with the isotropic scaling term.
4. `Translation`, a `d`-length vector representing the translation.
5. `KernelMatrix` of shape `(m,m)` — the Gaussian RBF Gram matrix between moving points.
6. `CoeffMatrix` of shape `(m,d)` — the GP coefficient matrix defining the displacement field.

In addition to `TransformParams`, `align` returns a tuple of the final variance and the number of iterations performed. `align_fixed_iter` instead returns an array of the variance at each step of the optimization.

Additional options:
- `inner_iter` (default: 2) — number of rigid↔non-rigid alternations inside each M-step.
- `burn_in` (default: 0) — number of initial EM iterations with `W=0` (pure rigid phase) before allowing non-rigid deformation.

::: cpdx.rnr.align
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.rnr.align_fixed_iter
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.rnr.transform
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.rnr.interpolate
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.rnr.maximization
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: cpdx.rnr.maximization_uncertainty
    handler: python
    options:
        show_source: false
        show_root_heading: true
