"""
k-NN based alignment (ICP iteration) between point clouds

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Optional, TYPE_CHECKING, Union, Tuple

from .. import _trueform
from .._dispatch import extract_meta, build_suffix

if TYPE_CHECKING:
    from .._spatial.point_cloud import PointCloud


def fit_knn_alignment(
    cloud0: Union["PointCloud", Tuple["PointCloud", np.ndarray]],
    cloud1: Union["PointCloud", Tuple["PointCloud", np.ndarray]],
    *,
    k: int = 1,
    sigma: Optional[float] = None,
    outlier_proportion: float = 0.0,
) -> np.ndarray:
    """
    Fit a rigid transformation using k-nearest neighbor correspondences.

    For each point in cloud0, finds the k nearest neighbors in cloud1 and
    computes a weighted correspondence point. The weights use a Gaussian kernel:
        weight_j = exp(-dist_j² / (2σ²))
    where σ defaults to the distance of the k-th neighbor (adaptive scaling).

    This is equivalent to one iteration of ICP when k=1. For k>1, soft
    correspondences provide robustness to noise and partial overlap.

    Returns a DELTA transformation mapping source world coordinates to target
    world coordinates. To get the total transformation for source local coords:

    >>> delta = tf.fit_knn_alignment(source, target)
    >>> total = delta @ source.transformation
    >>> source.transformation = total

    Supports three alignment modes based on input types:

    - **Point-to-Point**: Both inputs are PointCloud. Classic ICP step.
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
        Target point cloud (searched for neighbors), optionally with normals
    k : int, optional
        Number of nearest neighbors (default: 1 = classic ICP)
    sigma : float, optional
        Gaussian kernel width. If None, uses the k-th neighbor distance
        as sigma (adaptive). Default: None
    outlier_proportion : float, optional
        Proportion of worst correspondences to reject (0-1). Default: 0

    Returns
    -------
    transformation : ndarray of shape (3, 3) for 2D or (4, 4) for 3D
        Delta transformation mapping source_world -> target_world.

    Examples
    --------
    >>> import trueform as tf
    >>> # Point-to-point (single ICP step)
    >>> delta = tf.fit_knn_alignment(source, target, k=1)
    >>> source.transformation = delta @ source.transformation

    >>> # Point-to-plane (faster convergence)
    >>> delta = tf.fit_knn_alignment(source, (target, target_normals))

    >>> # With outlier rejection
    >>> delta = tf.fit_knn_alignment(source, target, k=5, outlier_proportion=0.1)
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
        func_name = f"fit_knn_alignment_weighted_{suffix}"
        cpp_func = getattr(_trueform.geometry, func_name)
        return cpp_func(c0._wrapper, normals0, c1._wrapper, normals1,
                        k, sigma, outlier_proportion)
    elif normals1 is not None:
        # Point-to-plane (target has normals)
        func_name = f"fit_knn_alignment_p2plane_{suffix}"
        cpp_func = getattr(_trueform.geometry, func_name)
        return cpp_func(c0._wrapper, c1._wrapper, normals1,
                        k, sigma, outlier_proportion)
    else:
        # Point-to-point
        func_name = f"fit_knn_alignment_{suffix}"
        cpp_func = getattr(_trueform.geometry, func_name)
        return cpp_func(c0._wrapper, c1._wrapper, k, sigma, outlier_proportion)
