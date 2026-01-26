"""
k-NN based alignment (ICP iteration) between point clouds

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Optional, TYPE_CHECKING

from .. import _trueform
from .._dispatch import extract_meta, build_suffix

if TYPE_CHECKING:
    from .._spatial.point_cloud import PointCloud


def fit_knn_alignment(
    cloud0: "PointCloud",
    cloud1: "PointCloud",
    k: int = 1,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """
    Fit a rigid transformation using k-nearest neighbor correspondences.

    For each point in cloud0, finds the k nearest neighbors in cloud1 and
    computes a weighted correspondence point. The weights use a Gaussian kernel:
        weight_j = exp(-dist_j² / (2σ²))
    where σ defaults to the distance of the k-th neighbor (adaptive scaling).

    This is equivalent to one iteration of ICP when k=1. For k>1, soft
    correspondences provide robustness to noise and partial overlap.

    If point clouds have transformations set, the alignment is computed
    in world space (with transformations applied).

    Parameters
    ----------
    cloud0 : PointCloud
        Source point cloud
    cloud1 : PointCloud
        Target point cloud (searched for neighbors)
    k : int, optional
        Number of nearest neighbors (default: 1 = classic ICP)
    sigma : float, optional
        Gaussian kernel width. If None, uses the k-th neighbor distance
        as sigma (adaptive). Default: None

    Returns
    -------
    transformation : ndarray of shape (3, 3) for 2D or (4, 4) for 3D
        Homogeneous transformation matrix mapping cloud0 -> cloud1

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> # ICP iteration
    >>> pts0 = np.random.randn(100, 3).astype(np.float32)
    >>> pts1 = pts0 @ rotation_matrix + translation + noise
    >>> cloud0 = tf.PointCloud(pts0)
    >>> cloud1 = tf.PointCloud(pts1)
    >>> # Single ICP iteration (k=1)
    >>> T = tf.fit_knn_alignment(cloud0, cloud1)
    >>> # Soft correspondences with 5 neighbors
    >>> T = tf.fit_knn_alignment(cloud0, cloud1, k=5)
    """
    if cloud0.dims != cloud1.dims:
        raise ValueError(
            f"Dimension mismatch: cloud0 has {cloud0.dims}D, cloud1 has {cloud1.dims}D"
        )
    if cloud0.dtype != cloud1.dtype:
        raise ValueError(
            f"Dtype mismatch: cloud0 has {cloud0.dtype}, cloud1 has {cloud1.dtype}"
        )

    func_name = f"fit_knn_alignment_{build_suffix(extract_meta(cloud0))}"
    cpp_func = getattr(_trueform.geometry, func_name)
    return cpp_func(cloud0._wrapper, cloud1._wrapper, k, sigma)
