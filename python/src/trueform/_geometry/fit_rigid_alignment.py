"""
Rigid alignment (Kabsch/Procrustes) between point clouds

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import TYPE_CHECKING, Union, Tuple

from .. import _trueform
from .._dispatch import extract_meta, build_suffix

if TYPE_CHECKING:
    from .._spatial.point_cloud import PointCloud


def fit_rigid_alignment(
    cloud0: Union["PointCloud", Tuple["PointCloud", np.ndarray]],
    cloud1: Union["PointCloud", Tuple["PointCloud", np.ndarray]],
) -> np.ndarray:
    """
    Fit a rigid transformation (rotation + translation) from cloud0 to cloud1.

    Computes the optimal rigid transformation using the Kabsch/Procrustes
    algorithm. Point clouds must have the same number of points and be in
    correspondence (point i in cloud0 corresponds to point i in cloud1).

    Returns a DELTA transformation mapping source world coordinates to target
    world coordinates. To get the total transformation for source local coords:

    >>> delta = tf.fit_rigid_alignment(source, target)
    >>> total = delta @ source.transformation
    >>> source.transformation = total

    Supports three alignment modes based on input types:

    - **Point-to-Point**: Both inputs are PointCloud. Classic Kabsch algorithm.
    - **Point-to-Plane**: cloud1 is (PointCloud, normals) tuple. Minimizes
      distance along normal direction. Converges faster.
    - **Normal Weighting**: Both are (PointCloud, normals) tuples. Weights
      correspondences by normal agreement for better accuracy.

    If point clouds have transformations set, the alignment is computed
    in world space (with transformations applied).

    Parameters
    ----------
    cloud0 : PointCloud or (PointCloud, normals)
        Source point cloud, optionally with normals for weighting
    cloud1 : PointCloud or (PointCloud, normals)
        Target point cloud, optionally with normals for point-to-plane

    Returns
    -------
    transformation : ndarray of shape (3, 3) for 2D or (4, 4) for 3D
        Delta transformation mapping source_world -> target_world.

    Examples
    --------
    >>> import trueform as tf
    >>> # Point-to-point alignment
    >>> delta = tf.fit_rigid_alignment(source, target)
    >>> source.transformation = delta @ source.transformation

    >>> # Point-to-plane (faster convergence)
    >>> delta = tf.fit_rigid_alignment(source, (target, target_normals))

    >>> # Normal weighting (best accuracy)
    >>> delta = tf.fit_rigid_alignment((source, src_normals), (target, tgt_normals))
    """
    # Extract clouds and normals from tuples
    if isinstance(cloud0, tuple):
        c0, normals0 = cloud0
    else:
        c0, normals0 = cloud0, None

    if isinstance(cloud1, tuple):
        c1, normals1 = cloud1
    else:
        c1, normals1 = cloud1, None

    if c0.dims != c1.dims:
        raise ValueError(
            f"Dimension mismatch: cloud0 has {c0.dims}D, cloud1 has {c1.dims}D"
        )
    if c0.dtype != c1.dtype:
        raise ValueError(
            f"Dtype mismatch: cloud0 has {c0.dtype}, cloud1 has {c1.dtype}"
        )
    if (normals0 is not None or normals1 is not None) and c0.dims != 3:
        raise ValueError(
            "Point-to-plane and normal weighting only supported for 3D point clouds"
        )

    suffix = build_suffix(extract_meta(c0))

    if normals0 is not None and normals1 is not None:
        # Normal weighting (both have normals)
        func_name = f"fit_rigid_alignment_weighted_{suffix}"
        cpp_func = getattr(_trueform.geometry, func_name)
        return cpp_func(c0._wrapper, normals0, c1._wrapper, normals1)
    elif normals1 is not None:
        # Point-to-plane (target has normals)
        func_name = f"fit_rigid_alignment_p2plane_{suffix}"
        cpp_func = getattr(_trueform.geometry, func_name)
        return cpp_func(c0._wrapper, c1._wrapper, normals1)
    else:
        # Point-to-point
        func_name = f"fit_rigid_alignment_{suffix}"
        cpp_func = getattr(_trueform.geometry, func_name)
        return cpp_func(c0._wrapper, c1._wrapper)
