"""
Chamfer error between point clouds

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import TYPE_CHECKING

from .. import _trueform
from .._dispatch import extract_meta, build_suffix

if TYPE_CHECKING:
    from .._spatial.point_cloud import PointCloud


def chamfer_error(cloud0: "PointCloud", cloud1: "PointCloud") -> float:
    """
    Compute one-way Chamfer error from cloud0 to cloud1.

    For each point in cloud0, finds the nearest point in cloud1 and
    accumulates the distance. Returns the mean distance. This is an
    asymmetric measure; for symmetric Chamfer distance, compute both
    directions and average.

    If point clouds have transformations set, the computation is
    performed in world space (with transformations applied).

    Parameters
    ----------
    cloud0 : PointCloud
        Source point cloud
    cloud1 : PointCloud
        Target point cloud

    Returns
    -------
    error : float
        Mean nearest-neighbor distance from cloud0 to cloud1

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> pts0 = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    >>> pts1 = np.array([[0, 0, 0.1], [1, 0, 0.1]], dtype=np.float32)
    >>> cloud0 = tf.PointCloud(pts0)
    >>> cloud1 = tf.PointCloud(pts1)
    >>> error = tf.chamfer_error(cloud0, cloud1)
    >>> # Symmetric Chamfer distance
    >>> symmetric_error = (tf.chamfer_error(cloud0, cloud1) +
    ...                    tf.chamfer_error(cloud1, cloud0)) / 2
    """
    if cloud0.dims != cloud1.dims:
        raise ValueError(
            f"Dimension mismatch: cloud0 has {cloud0.dims}D, cloud1 has {cloud1.dims}D"
        )
    if cloud0.dtype != cloud1.dtype:
        raise ValueError(
            f"Dtype mismatch: cloud0 has {cloud0.dtype}, cloud1 has {cloud1.dtype}"
        )

    func_name = f"chamfer_error_{build_suffix(extract_meta(cloud0))}"
    cpp_func = getattr(_trueform.geometry, func_name)
    return cpp_func(cloud0._wrapper, cloud1._wrapper)
