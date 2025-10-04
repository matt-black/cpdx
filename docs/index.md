# cpdx

Point cloud registration with coherent point drift.
The implementation is in [JAX](https://docs.jax.dev/en/latest/) and is thus GPU/TPU/CPU-native.

## Introduction

Coherent point drift (CPD, [1,2]) and its associated bayesian variant (BCPD, [3,4]) are popular algorithms for point cloud registration.

This library provides algorithms for rigid, affine, and non-rigid ("deformable") point cloud registration with coherent point drift. Each of these options is available under `cpdx.rigid`, `cpdx.affine` and `cpdx.deformable` modules, respectively. All of these modules expose the same API:

- `align`: optimize the alignment up to a specified tolerance, or a maximum number of iterations (whichever happens first).
- `align_fixed_iter`: optimize the alignment for a fixed number of EM iterations.
- `transform`: transform a set of points using the specified transform.
- `maximization`: do a single M-step of the CPD algorithm.

An implementation of bayesian coherent point drift is available in the `cpdx.bayes` module.

## References

[1] Myronenko, Andriy, Xubo Song, and Miguel Carreira-Perpinan. "Non-rigid point set registration: Coherent point drift." Advances in neural information processing systems 19 (2006).

[2] Myronenko, Andriy, and Xubo Song. "Point set registration: Coherent point drift." IEEE transactions on pattern analysis and machine intelligence 32.12 (2010): 2262-2275.

[3] Hirose, Osamu. "A Bayesian formulation of coherent point drift." IEEE transactions on pattern analysis and machine intelligence 43.7 (2020): 2269-2286.

[4] Hirose, Osamu. "Acceleration of non-rigid point set registration with downsampling and Gaussian process regression." IEEE Transactions on Pattern Analysis and Machine Intelligence 43.8 (2020): 2858-2865.
