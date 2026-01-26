"""
OBB-based alignment between point clouds

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import TYPE_CHECKING

from .. import _trueform
from .._dispatch import extract_meta, build_suffix

if TYPE_CHECKING:
    from .._spatial.point_cloud import PointCloud


def fit_obb_alignment(
    cloud0: "PointCloud", cloud1: "PointCloud", sample_size: int = 100
) -> np.ndarray:
    """
    Compute a rigid alignment from cloud0 to cloud1 using oriented bounding boxes.

    The returned transform T maps points from cloud0 into cloud1:
        y ≈ T(x) = R * x + t
    where R aligns the OBB axes of cloud0 to those of cloud1, and t aligns
    the OBB centers. No point correspondences are used.

    If point clouds have transformations set, the alignment is computed
    in world space (with transformations applied).

    OBB alignment is inherently ambiguous up to the symmetry group of the
    bounding box (180° rotations about each axis). The function resolves
    this ambiguity by testing orientations and selecting the one with
    lowest chamfer distance using sampled points.

    Parameters
    ----------
    cloud0 : PointCloud
        Source point cloud
    cloud1 : PointCloud
        Target point cloud
    sample_size : int, optional
        Number of points to sample for disambiguation (default: 100)

    Returns
    -------
    transformation : ndarray of shape (3, 3) for 2D or (4, 4) for 3D
        Homogeneous transformation matrix mapping cloud0 -> cloud1

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> # Create point clouds with different orientations
    >>> pts0 = np.random.randn(100, 3).astype(np.float32)
    >>> pts1 = pts0 @ rotation_matrix + translation
    >>> cloud0 = tf.PointCloud(pts0)
    >>> cloud1 = tf.PointCloud(pts1)
    >>> T = tf.fit_obb_alignment(cloud0, cloud1)
    """
    if cloud0.dims != cloud1.dims:
        raise ValueError(
            f"Dimension mismatch: cloud0 has {cloud0.dims}D, cloud1 has {cloud1.dims}D"
        )
    if cloud0.dtype != cloud1.dtype:
        raise ValueError(
            f"Dtype mismatch: cloud0 has {cloud0.dtype}, cloud1 has {cloud1.dtype}"
        )

    func_name = f"fit_obb_alignment_{build_suffix(extract_meta(cloud0))}"
    cpp_func = getattr(_trueform.geometry, func_name)
    return cpp_func(cloud0._wrapper, cloud1._wrapper, sample_size)
