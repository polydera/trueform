"""
Chamfer error between point clouds

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import TYPE_CHECKING, Union
import numpy as np

from .. import _trueform
from .._dispatch import extract_meta, build_suffix, ensure_point_cloud

if TYPE_CHECKING:
    from .._spatial.point_cloud import PointCloud


def chamfer_error(
    source: Union["PointCloud", np.ndarray, tuple],
    target: "PointCloud"
) -> float:
    """
    Compute one-way Chamfer error from source to target.

    For each point in source, finds the nearest point in target and
    accumulates the distance. Returns the mean distance. This is an
    asymmetric measure; for symmetric Chamfer distance, compute both
    directions and average.

    If point clouds have transformations set, the computation is
    performed in world space (with transformations applied).

    Parameters
    ----------
    source : PointCloud, ndarray, or tuple
        Source points. Can be:
        - PointCloud object
        - ndarray of shape (N, dims) - points array
        - tuple (points_array, transformation_matrix)
    target : PointCloud
        Target point cloud (tree auto-built on first access)

    Returns
    -------
    error : float
        Mean nearest-neighbor distance from source to target

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> pts0 = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    >>> pts1 = np.array([[0, 0, 0.1], [1, 0, 0.1]], dtype=np.float32)
    >>> cloud0 = tf.PointCloud(pts0)
    >>> cloud1 = tf.PointCloud(pts1)
    >>> error = tf.chamfer_error(cloud0, cloud1)
    >>>
    >>> # With subsampling (every 10th point)
    >>> error = tf.chamfer_error(pts0[::10], cloud1)
    >>>
    >>> # With transformation
    >>> T = np.eye(4, dtype=np.float32)
    >>> error = tf.chamfer_error((pts0, T), cloud1)
    >>>
    >>> # Symmetric Chamfer distance
    >>> symmetric_error = (tf.chamfer_error(cloud0, cloud1) +
    ...                    tf.chamfer_error(cloud1, cloud0)) / 2
    """
    source = ensure_point_cloud(source)
    target = ensure_point_cloud(target, dims=source.dims)

    if source.dtype != target.dtype:
        raise ValueError(
            f"Dtype mismatch: source has {source.dtype}, target has {target.dtype}"
        )

    func_name = f"chamfer_error_{build_suffix(extract_meta(source))}"
    cpp_func = getattr(_trueform.geometry, func_name)
    return cpp_func(source._wrapper, target._wrapper)
