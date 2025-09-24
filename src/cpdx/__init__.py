from typing import Union

from jaxtyping import Array
from jaxtyping import Float

from .affine import TransformParams as AffineTransformParams
from .affine import align as align_affine
from .affine import align_fixed_iter as align_fixed_iter_affine
from .deformable import TransformParams as DeformableTransformParams
from .deformable import align as align_deformable
from .deformable import align_fixed_iter as align_fixed_iter_deformable
from .rigid import TransformParams as RigidTransformParams
from .rigid import align as align_rigid
from .rigid import align_fixed_iter as align_fixed_iter_rigid


type TransformParams = Union[
    RigidTransformParams, AffineTransformParams, DeformableTransformParams
]


__all__ = [
    "align",
]


def align(
    ref: Float[Array, "n d"],
    mov: Float[Array, "m d"],
    method: str,
    outlier_prob: float,
    num_iter: int,
    tolerance: float | None,
    regularization_param_deformable: float = 1.0,
    kernel_stddev_deformable: float = 1.0,
) -> tuple[
    TransformParams, Union[Float[Array, " d"], tuple[Float[Array, ""], int]]
]:
    if method == "deformable":
        if tolerance is None:
            return align_fixed_iter_deformable(
                ref,
                mov,
                outlier_prob,
                regularization_param_deformable,
                kernel_stddev_deformable,
                num_iter,
            )
        else:
            return align_deformable(
                ref,
                mov,
                outlier_prob,
                regularization_param_deformable,
                kernel_stddev_deformable,
                num_iter,
                tolerance,
            )
    elif method == "affine":
        if tolerance is None:
            return align_fixed_iter_affine(ref, mov, outlier_prob, num_iter)
        else:
            return align_affine(ref, mov, outlier_prob, num_iter, tolerance)
    elif method == "rigid":
        if tolerance is None:
            return align_fixed_iter_rigid(ref, mov, outlier_prob, num_iter)
        else:
            return align_rigid(ref, mov, outlier_prob, num_iter, tolerance)
    else:
        raise ValueError("invalid method")
