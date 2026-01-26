"""
Closest metric point pair functions

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Any, Tuple
from .. import _trueform
from .._primitives import Point, Segment, Polygon, Ray, Line, Plane
from .._dispatch import InputMeta, build_suffix


# Dispatch table for closest_metric_point_pair
# Maps (type0, type1) -> (function_name_template, needs_swap)
# needs_swap=True means we need to swap arguments and results
_CLOSEST_PAIR_DISPATCH = {
    (Point, Point): ("closest_metric_point_pair_point_point_{}", False),
    (Point, Segment): ("closest_metric_point_pair_point_segment_{}", False),
    (Segment, Point): ("closest_metric_point_pair_point_segment_{}", True),
    (Point, Polygon): ("closest_metric_point_pair_point_polygon_{}", False),
    (Polygon, Point): ("closest_metric_point_pair_point_polygon_{}", True),
    (Segment, Segment): ("closest_metric_point_pair_segment_segment_{}", False),
    (Segment, Polygon): ("closest_metric_point_pair_segment_polygon_{}", False),
    (Polygon, Segment): ("closest_metric_point_pair_segment_polygon_{}", True),
    (Polygon, Polygon): ("closest_metric_point_pair_polygon_polygon_{}", False),
    (Point, Ray): ("closest_metric_point_pair_point_ray_{}", False),
    (Ray, Point): ("closest_metric_point_pair_point_ray_{}", True),
    (Point, Line): ("closest_metric_point_pair_point_line_{}", False),
    (Line, Point): ("closest_metric_point_pair_point_line_{}", True),
    (Segment, Ray): ("closest_metric_point_pair_segment_ray_{}", False),
    (Ray, Segment): ("closest_metric_point_pair_segment_ray_{}", True),
    (Segment, Line): ("closest_metric_point_pair_segment_line_{}", False),
    (Line, Segment): ("closest_metric_point_pair_segment_line_{}", True),
    (Polygon, Ray): ("closest_metric_point_pair_polygon_ray_{}", False),
    (Ray, Polygon): ("closest_metric_point_pair_polygon_ray_{}", True),
    (Polygon, Line): ("closest_metric_point_pair_polygon_line_{}", False),
    (Line, Polygon): ("closest_metric_point_pair_polygon_line_{}", True),
    (Ray, Ray): ("closest_metric_point_pair_ray_ray_{}", False),
    (Line, Line): ("closest_metric_point_pair_line_line_{}", False),
    (Ray, Line): ("closest_metric_point_pair_ray_line_{}", False),
    (Line, Ray): ("closest_metric_point_pair_ray_line_{}", True),
    # Plane combinations (3D only)
    (Point, Plane): ("closest_metric_point_pair_point_plane_{}", False),
    (Plane, Point): ("closest_metric_point_pair_plane_point_{}", False),
    (Segment, Plane): ("closest_metric_point_pair_segment_plane_{}", False),
    (Plane, Segment): ("closest_metric_point_pair_plane_segment_{}", False),
    (Ray, Plane): ("closest_metric_point_pair_ray_plane_{}", False),
    (Plane, Ray): ("closest_metric_point_pair_plane_ray_{}", False),
    (Line, Plane): ("closest_metric_point_pair_line_plane_{}", False),
    (Plane, Line): ("closest_metric_point_pair_plane_line_{}", False),
    (Polygon, Plane): ("closest_metric_point_pair_polygon_plane_{}", False),
    (Plane, Polygon): ("closest_metric_point_pair_plane_polygon_{}", False),
    (Plane, Plane): ("closest_metric_point_pair_plane_plane_{}", False),
}


def closest_metric_point(obj0: Any, obj1: Any) -> Tuple[float, np.ndarray]:
    """
    Compute closest point on obj0 to obj1.

    Returns the squared distance and the closest point on obj0.

    Parameters
    ----------
    obj0, obj1
        Geometric objects (Point, Segment, Polygon, Ray, Line, Plane, or numpy arrays)

    Returns
    -------
    distance_squared : float
        Squared distance between obj0 and obj1
    closest_point : np.ndarray
        Closest point on obj0 (first argument)

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> seg = tf.Segment([[1, 0, 0], [1, 1, 0]])
    >>> pt = tf.Point([0, 0, 0])
    >>> dist2, closest = tf.closest_metric_point(seg, pt)
    >>> dist2
    1.0
    >>> closest
    array([1., 0., 0.])
    """
    dist2, closest_pt, _ = closest_metric_point_pair(obj0, obj1)
    return dist2, closest_pt


def closest_metric_point_pair(obj0: Any, obj1: Any) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute closest point pair between two geometric objects.

    Returns the squared distance and the two closest points, one on each object.

    Parameters
    ----------
    obj0, obj1
        Geometric objects (Point, Segment, Polygon, Ray, Line, or numpy arrays)

    Returns
    -------
    distance_squared : float
        Squared distance between the objects
    point0 : np.ndarray
        Closest point on obj0
    point1 : np.ndarray
        Closest point on obj1

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> pt0 = tf.Point([0, 0, 0])
    >>> pt1 = tf.Point([1, 1, 1])
    >>> dist2, p0, p1 = tf.closest_metric_point_pair(pt0, pt1)
    >>> dist2
    3.0
    """

    # Helper to get dimensionality
    def get_dims(obj):
        if hasattr(obj, 'dims'):
            return obj.dims
        elif isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                return obj.shape[0]
            elif obj.ndim == 2:
                return obj.shape[1]
        raise TypeError(f"Cannot determine dimensions for type {type(obj)}")

    # Helper to get variant suffix
    def get_suffix(obj):
        if hasattr(obj, 'dtype') and hasattr(obj, 'dims'):
            meta = InputMeta(None, obj.dtype, None, obj.dims)
            return build_suffix(meta)
        elif isinstance(obj, np.ndarray):
            # For raw arrays, infer dims from shape
            dims = obj.shape[0] if obj.ndim == 1 else obj.shape[1]
            meta = InputMeta(None, obj.dtype, None, dims)
            return build_suffix(meta)
        raise TypeError(f"Cannot determine variant for type {type(obj)}")

    # Helper to extract data from object (handles Point wrappers and numpy arrays)
    def get_data(obj):
        if hasattr(obj, 'data'):
            return obj.data
        elif isinstance(obj, np.ndarray):
            return obj
        raise TypeError(f"Cannot extract data from type {type(obj)}")

    # Validate dimensions match
    dims0 = get_dims(obj0)
    dims1 = get_dims(obj1)
    if dims0 != dims1:
        raise ValueError(
            f"Dimension mismatch: obj0 has {dims0}D, obj1 has {dims1}D. "
            f"Both objects must have the same dimensionality (2D or 3D)."
        )

    # Normalize types (treat numpy arrays as Point for dispatch)
    type0 = Point if isinstance(
        obj0, np.ndarray) and obj0.ndim == 1 else type(obj0)
    type1 = Point if isinstance(
        obj1, np.ndarray) and obj1.ndim == 1 else type(obj1)

    # Look up dispatch info
    type_pair = (type0, type1)
    if type_pair not in _CLOSEST_PAIR_DISPATCH:
        raise TypeError(
            f"closest_metric_point_pair not implemented for types: "
            f"{type0.__name__}, {type1.__name__}"
        )

    func_template, needs_swap = _CLOSEST_PAIR_DISPATCH[type_pair]

    # Special case: Plane is 3D only
    if (type0 is Plane or type1 is Plane) and dims0 != 3:
        raise ValueError("closest_metric_point_pair with Plane is only supported in 3D")

    # Get suffix and function name
    suffix = get_suffix(obj0 if not needs_swap else obj1)
    func_name = func_template.format(suffix)

    # Get data and call C++ function
    data0 = get_data(obj0)
    data1 = get_data(obj1)

    if needs_swap:
        # Swap arguments and results
        dist2, pt1, pt0 = getattr(_trueform.core, func_name)(data1, data0)
        return dist2, pt0, pt1
    else:
        return getattr(_trueform.core, func_name)(data0, data1)
