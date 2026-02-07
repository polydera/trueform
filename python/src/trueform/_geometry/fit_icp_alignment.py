"""
ICP (Iterative Closest Point) alignment between point clouds

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


def fit_icp_alignment(
    cloud0: Union["PointCloud", Tuple["PointCloud", np.ndarray]],
    cloud1: Union["PointCloud", Tuple["PointCloud", np.ndarray]],
    *,
    max_iterations: int = 100,
    n_samples: int = 1000,
    k: int = 1,
    sigma: Optional[float] = None,
    outlier_proportion: float = 0.0,
    min_relative_improvement: float = 1e-6,
    ema_alpha: float = 0.3,
) -> np.ndarray:
    """
    Fit a rigid transformation using Iterative Closest Point (ICP).

    Iteratively refines alignment by finding correspondences and fitting
    rigid transformations. Includes automatic convergence detection.

    Supports three alignment modes based on input types:

    - **Point-to-Point**: Both inputs are PointCloud. Classic ICP.
    - **Point-to-Plane**: cloud1 is (PointCloud, normals) tuple. Minimizes
      distance along normal direction. Converges 2-3x faster.
    - **Normal Weighting**: Both are (PointCloud, normals) tuples. Weights
      correspondences by normal agreement for best accuracy.

    Returns a DELTA transformation mapping source world coordinates to target
    world coordinates. To get the total transformation for source local coords:

    >>> delta = tf.fit_icp_alignment(source, target)
    >>> total = delta @ source.transformation
    >>> source.transformation = total

    Parameters
    ----------
    cloud0 : PointCloud or (PointCloud, normals)
        Source point cloud, optionally with normals for weighting
    cloud1 : PointCloud or (PointCloud, normals)
        Target point cloud, optionally with normals for point-to-plane
    max_iterations : int, optional
        Maximum number of ICP iterations (default: 100)
    n_samples : int, optional
        Number of points to subsample per iteration (default: 1000, 0=all)
    k : int, optional
        Number of nearest neighbors (default: 1 = classic ICP)
    sigma : float, optional
        Gaussian kernel width. If None, uses adaptive scaling
    outlier_proportion : float, optional
        Proportion of worst correspondences to reject (0-1). Default: 0
    min_relative_improvement : float, optional
        Convergence threshold (default: 1e-6)
    ema_alpha : float, optional
        EMA smoothing factor for convergence detection (default: 0.3)

    Returns
    -------
    transformation : ndarray of shape (4, 4) for 3D or (3, 3) for 2D
        Delta transformation mapping source_world -> target_world.

    Examples
    --------
    >>> import trueform as tf
    >>> # Point-to-point ICP
    >>> delta = tf.fit_icp_alignment(source, target, max_iterations=50)

    >>> # Point-to-plane ICP (faster convergence)
    >>> delta = tf.fit_icp_alignment(source, (target, target_normals))

    >>> # With initial alignment - compose delta with source frame
    >>> source.transformation = T_initial
    >>> delta = tf.fit_icp_alignment(source, target)
    >>> total = delta @ T_initial
    >>> source.transformation = total
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

    # Convert sigma to float, using -1 for adaptive
    sigma_val = float(sigma) if sigma is not None else None

    if normals0 is not None and normals1 is not None:
        # Normal weighting (both have normals)
        func_name = f"fit_icp_alignment_weighted_{suffix}"
        cpp_func = getattr(_trueform.geometry, func_name)
        return cpp_func(
            c0._wrapper, normals0, c1._wrapper, normals1,
            max_iterations, n_samples, k, sigma_val,
            outlier_proportion, min_relative_improvement, ema_alpha
        )
    elif normals1 is not None:
        # Point-to-plane (target has normals)
        func_name = f"fit_icp_alignment_p2plane_{suffix}"
        cpp_func = getattr(_trueform.geometry, func_name)
        return cpp_func(
            c0._wrapper, c1._wrapper, normals1,
            max_iterations, n_samples, k, sigma_val,
            outlier_proportion, min_relative_improvement, ema_alpha
        )
    else:
        # Point-to-point
        func_name = f"fit_icp_alignment_{suffix}"
        cpp_func = getattr(_trueform.geometry, func_name)
        return cpp_func(
            c0._wrapper, c1._wrapper,
            max_iterations, n_samples, k, sigma_val,
            outlier_proportion, min_relative_improvement, ema_alpha
        )
