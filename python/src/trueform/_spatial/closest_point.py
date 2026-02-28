"""
Closest metric point pair API

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Any, Tuple
import numpy as np
from .. import _trueform
from .._primitives.primitive import Primitive
from .._dispatch import InputMeta, build_suffix


def closest_metric_point_pair(obj0: Any, obj1: Any) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the closest point pair and squared distance between two primitives.

    Finds the pair of points (one on each object) that minimizes the distance,
    and returns the squared distance metric along with both points.
    Supports batch broadcasting.

    Parameters
    ----------
    obj0, obj1 : Primitive
        Any combination of Point, Segment, Triangle, Polygon, Ray, Line, Plane, AABB.

    Returns
    -------
    metric : float or np.ndarray
        Squared distance between the closest points.
        Scalar for single x single, ndarray for batch.
    point0 : np.ndarray
        Closest point on obj0. Shape (D,) for single, (N, D) for batch.
    point1 : np.ndarray
        Closest point on obj1. Shape (D,) for single, (N, D) for batch.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> # Closest points between two segments
    >>> seg1 = tf.Segment(start=[0, 0, 0], end=[1, 0, 0])
    >>> seg2 = tf.Segment(start=[0, 1, 0], end=[1, 1, 0])
    >>> metric, pt0, pt1 = tf.closest_metric_point_pair(seg1, seg2)

    >>> # Closest point from a point to a segment
    >>> pt = tf.Point([0.5, 0.5, 0.0])
    >>> seg = tf.Segment(start=[0, 0, 0], end=[1, 0, 0])
    >>> metric, closest_on_pt, closest_on_seg = tf.closest_metric_point_pair(pt, seg)
    """
    if not isinstance(obj0, Primitive) or not isinstance(obj1, Primitive):
        raise TypeError(
            f"closest_metric_point_pair requires two Primitives, got "
            f"{type(obj0).__name__} and {type(obj1).__name__}")

    if obj0.dims != obj1.dims:
        raise ValueError(
            f"Dimension mismatch: obj0 has {obj0.dims}D, obj1 has {obj1.dims}D. "
            f"Both objects must have the same dimensionality (2D or 3D).")

    if obj0.dtype != obj1.dtype:
        raise TypeError(
            f"Dtype mismatch: obj0 has {obj0.dtype}, obj1 has {obj1.dtype}. "
            f"Both objects must have the same dtype.")

    suffix = build_suffix(InputMeta(None, obj0.dtype, None, obj0.dims))
    fn = getattr(_trueform.spatial, f"closest_point_pair_pp_{suffix}")
    return fn(obj0._wrapper, obj1._wrapper)


def closest_metric_point(obj0: Any, obj1: Any) -> Tuple[float, np.ndarray]:
    """
    Compute closest point on obj0 to obj1.

    Returns the squared distance and the closest point on obj0.

    Parameters
    ----------
    obj0, obj1 : Primitive
        Any combination of Point, Segment, Triangle, Polygon, Ray, Line, Plane, AABB.

    Returns
    -------
    distance_squared : float
        Squared distance between obj0 and obj1
    closest_point : np.ndarray
        Closest point on obj0 (first argument)
    """
    dist2, closest_pt, _ = closest_metric_point_pair(obj0, obj1)
    return dist2, closest_pt


# Alias
closest_point_pair = closest_metric_point_pair
