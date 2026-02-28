"""
Unified distance and distance2 API

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Any
from .. import _trueform
from .._primitives.primitive import Primitive
from .._dispatch import InputMeta, extract_meta, build_suffix


def distance(obj0: Any, obj1: Any):
    """
    Compute the Euclidean distance between two geometric objects.

    This function works with both core primitives (Point, Segment, Triangle, Polygon,
    Line, AABB, Ray, Plane) and spatial data structures (Mesh, EdgeMesh, PointCloud).
    Supports batch broadcasting: single x single returns a scalar, batch x single,
    single x batch, or batch x batch returns an ndarray.

    Parameters
    ----------
    obj0, obj1 : geometric objects
        Any combination of Point, Segment, Triangle, Polygon, Line, AABB, Ray, Plane,
        Mesh, EdgeMesh, PointCloud.

    Returns
    -------
    float or np.ndarray
        Distance between the objects. Scalar for single x single,
        ndarray for batch operations.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> # Distance from point to AABB
    >>> pt = tf.Point([0.5, 0.5])
    >>> box = tf.AABB(min=[1.0, 1.0], max=[2.0, 2.0])
    >>> tf.distance(pt, box)
    0.7071067811865476

    >>> # Distance between two segments
    >>> seg1 = tf.Segment([[0, 0], [1, 0]])
    >>> seg2 = tf.Segment([[0, 2], [1, 2]])
    >>> tf.distance(seg1, seg2)
    2.0

    >>> # Batch: distance from many points to a segment
    >>> pts = tf.Point(np.random.rand(100, 3).astype(np.float32))
    >>> seg = tf.Segment(start=[0, 0, 0], end=[1, 1, 1])
    >>> dists = tf.distance(pts, seg)  # shape (100,)
    """
    return _distance_impl(obj0, obj1, "distance")


def distance2(obj0: Any, obj1: Any):
    """
    Compute the squared Euclidean distance between two geometric objects.

    This is more efficient than distance() when you only need to compare distances,
    as it avoids the square root computation.

    Parameters
    ----------
    obj0, obj1 : geometric objects
        Any combination of Point, Segment, Triangle, Polygon, Line, AABB, Ray, Plane,
        Mesh, EdgeMesh, PointCloud.

    Returns
    -------
    float or np.ndarray
        Squared distance between the objects.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> # Squared distance from point to AABB
    >>> pt = tf.Point([0.5, 0.5])
    >>> box = tf.AABB(min=[1.0, 1.0], max=[2.0, 2.0])
    >>> tf.distance2(pt, box)
    0.5

    >>> # Squared distance between two points
    >>> pt1 = tf.Point([0, 0, 0])
    >>> pt2 = tf.Point([1, 0, 0])
    >>> tf.distance2(pt1, pt2)
    1.0
    """
    return _distance_impl(obj0, obj1, "distance2")


def _distance_impl(obj0: Any, obj1: Any, op: str):
    from . import Mesh, EdgeMesh, PointCloud

    is_prim0 = isinstance(obj0, Primitive)
    is_prim1 = isinstance(obj1, Primitive)
    is_form0 = isinstance(obj0, (Mesh, EdgeMesh, PointCloud))
    is_form1 = isinstance(obj1, (Mesh, EdgeMesh, PointCloud))

    if obj0.dims != obj1.dims:
        raise ValueError(
            f"Dimension mismatch: obj0 has {obj0.dims}D, obj1 has {obj1.dims}D. "
            f"Both objects must have the same dimensionality (2D or 3D).")

    if obj0.dtype != obj1.dtype:
        raise TypeError(
            f"Dtype mismatch: obj0 has {obj0.dtype}, obj1 has {obj1.dtype}. "
            f"Both objects must have the same dtype.")

    if is_prim0 and is_prim1:
        suffix = build_suffix(InputMeta(None, obj0.dtype, None, obj0.dims))
        fn = getattr(_trueform.spatial, f"{op}_pp_{suffix}")
        return fn(obj0._wrapper, obj1._wrapper)

    if is_form0 and is_prim1:
        meta = extract_meta(obj0)
        suffix = build_suffix(meta)
        fn = getattr(_trueform.spatial, f"{op}_{meta.form_name}_fp_{suffix}")
        return fn(obj0._wrapper, obj1._wrapper)

    if is_prim0 and is_form1:
        meta = extract_meta(obj1)
        suffix = build_suffix(meta)
        fn = getattr(_trueform.spatial, f"{op}_{meta.form_name}_fp_{suffix}")
        return fn(obj1._wrapper, obj0._wrapper)

    if is_form0 and is_form1:
        from .neighbor_search import neighbor_search as _neighbor_search
        result = _neighbor_search(obj0, obj1, radius=None)
        metric = result[1][0]
        return np.sqrt(metric) if op == "distance" else metric

    raise TypeError(
        f"{op} not supported for types: {type(obj0).__name__}, {type(obj1).__name__}")
