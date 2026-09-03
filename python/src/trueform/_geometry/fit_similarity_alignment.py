"""
Similarity alignment (Procrustes with scaling) between point clouds

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


def fit_similarity_alignment(
    cloud0: "PointCloud",
    cloud1: "PointCloud",
) -> np.ndarray:
    """
    Fit a similarity transformation (rotation + uniform scale + translation)
    from cloud0 to cloud1.

    Computes the optimal similarity transformation using the Procrustes
    algorithm with scaling. Point clouds must have the same number of points
    and be in correspondence (point i in cloud0 corresponds to point i in
    cloud1). The linear part of the returned matrix stores ``s * R``, so
    ``y ≈ s * R * x + t``.

    Returns a DELTA transformation mapping source world coordinates to target
    world coordinates. To get the total transformation for source local coords:

    >>> delta = tf.fit_similarity_alignment(source, target)
    >>> total = delta @ source.transformation
    >>> source.transformation = total

    If point clouds have transformations set, the alignment is computed
    in world space (with transformations applied) — the spread of the source
    is measured there too, so a frame that scales does not scale the answer.

    Parameters
    ----------
    cloud0 : PointCloud
        Source point cloud
    cloud1 : PointCloud
        Target point cloud

    Returns
    -------
    transformation : ndarray of shape (3, 3) for 2D or (4, 4) for 3D
        Delta transformation mapping source_world -> target_world; its
        linear part carries the uniform scale.

    Examples
    --------
    >>> import trueform as tf
    >>> delta = tf.fit_similarity_alignment(source, target)
    >>> source.transformation = delta @ source.transformation
    """
    if cloud0.dims != cloud1.dims:
        raise ValueError(
            f"Dimension mismatch: cloud0 has {cloud0.dims}D, "
            f"cloud1 has {cloud1.dims}D"
        )
    if cloud0.dtype != cloud1.dtype:
        raise ValueError(
            f"Dtype mismatch: cloud0 has {cloud0.dtype}, cloud1 has {cloud1.dtype}"
        )
    if cloud0.size != cloud1.size:
        raise ValueError(
            f"Point count mismatch: cloud0 has {cloud0.size} points, "
            f"cloud1 has {cloud1.size}"
        )

    suffix = build_suffix(extract_meta(cloud0))
    func_name = f"fit_similarity_alignment_{suffix}"
    cpp_func = getattr(_trueform.geometry, func_name)
    return cpp_func(cloud0._wrapper, cloud1._wrapper)
