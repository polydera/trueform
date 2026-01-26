"""
Rigid alignment (Kabsch/Procrustes) between point clouds

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


def fit_rigid_alignment(cloud0: "PointCloud", cloud1: "PointCloud") -> np.ndarray:
    """
    Fit a rigid transformation (rotation + translation) from cloud0 to cloud1.

    Computes the optimal rigid transformation T such that T(cloud0) ≈ cloud1
    using the Kabsch/Procrustes algorithm. Point clouds must have the same
    number of points and be in correspondence (point i in cloud0 corresponds
    to point i in cloud1).

    If point clouds have transformations set, the alignment is computed
    in world space (with transformations applied).

    Parameters
    ----------
    cloud0 : PointCloud
        Source point cloud
    cloud1 : PointCloud
        Target point cloud (must have same size as cloud0)

    Returns
    -------
    transformation : ndarray of shape (3, 3) for 2D or (4, 4) for 3D
        Homogeneous transformation matrix that best aligns cloud0 to cloud1

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> # Create corresponding point clouds
    >>> pts0 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    >>> pts1 = pts0 + [1, 2, 3]  # translated
    >>> cloud0 = tf.PointCloud(pts0)
    >>> cloud1 = tf.PointCloud(pts1)
    >>> T = tf.fit_rigid_alignment(cloud0, cloud1)
    >>> # T @ [x, y, z, 1] transforms points from cloud0 to cloud1
    """
    if cloud0.dims != cloud1.dims:
        raise ValueError(
            f"Dimension mismatch: cloud0 has {cloud0.dims}D, cloud1 has {cloud1.dims}D"
        )
    if cloud0.dtype != cloud1.dtype:
        raise ValueError(
            f"Dtype mismatch: cloud0 has {cloud0.dtype}, cloud1 has {cloud1.dtype}"
        )

    func_name = f"fit_rigid_alignment_{build_suffix(extract_meta(cloud0))}"
    cpp_func = getattr(_trueform.geometry, func_name)
    return cpp_func(cloud0._wrapper, cloud1._wrapper)
