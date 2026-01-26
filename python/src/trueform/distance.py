"""
Unified distance and distance2 API

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Any
from . import _trueform
from ._primitives import Plane

# Dispatch infrastructure
from ._dispatch import InputMeta, build_suffix
from ._core._dispatch import DISTANCE as CORE_DISTANCE


def distance(obj0: Any, obj1: Any) -> float:
    """
    Compute the Euclidean distance between two geometric objects.

    This function works with both core primitives (Point, Segment, Polygon, Line, AABB, Ray, Plane)
    and spatial data structures (Mesh, PointCloud - when spatial module is available).

    Parameters
    ----------
    obj0, obj1 : geometric objects
        Any combination of Point, Segment, Polygon, Line, AABB, Ray, Plane, Mesh, PointCloud, etc.

    Returns
    -------
    float
        Distance between the objects

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
    """
    return _distance_impl(obj0, obj1, "distance")


def distance2(obj0: Any, obj1: Any) -> float:
    """
    Compute the squared Euclidean distance between two geometric objects.

    This is more efficient than distance() when you only need to compare distances,
    as it avoids the square root computation.

    Parameters
    ----------
    obj0, obj1 : geometric objects
        Any combination of Point, Segment, Polygon, Line, AABB, Ray, Plane, Mesh, PointCloud, etc.

    Returns
    -------
    float
        Squared distance between the objects

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


def _distance_impl(obj0: Any, obj1: Any, func_prefix: str) -> float:
    """
    Internal implementation for both distance and distance2.

    Parameters
    ----------
    obj0, obj1 : geometric objects
    func_prefix : str
        Either "distance" or "distance2"
    """
    # Validate dimensions match
    if not hasattr(obj0, 'dims') or not hasattr(obj1, 'dims'):
        raise TypeError("Both objects must have 'dims' attribute")

    if obj0.dims != obj1.dims:
        raise ValueError(
            f"Dimension mismatch: obj0 has {obj0.dims}D, obj1 has {obj1.dims}D. "
            f"Both objects must have the same dimensionality (2D or 3D)."
        )

    # Check if either argument is a form (Mesh, EdgeMesh, PointCloud)
    from ._spatial import Mesh, EdgeMesh, PointCloud

    is_form0 = isinstance(obj0, (Mesh, EdgeMesh, PointCloud))
    is_form1 = isinstance(obj1, (Mesh, EdgeMesh, PointCloud))

    if is_form0 or is_form1:
        return _spatial_distance(obj0, obj1, is_form0, is_form1, func_prefix)
    else:
        return _core_distance(obj0, obj1, func_prefix)


def _core_distance(obj0, obj1, func_prefix: str) -> float:
    """Primitive x primitive distance."""
    type0 = type(obj0)
    type1 = type(obj1)
    type_pair = (type0, type1)

    if type_pair not in CORE_DISTANCE:
        supported = {t.__name__ for pair in CORE_DISTANCE.keys() for t in pair}
        raise TypeError(
            f"{func_prefix} not implemented for types: {type0.__name__}, {type1.__name__}. "
            f"Supported types: {', '.join(sorted(supported))}"
        )

    func_template, needs_swap = CORE_DISTANCE[type_pair]

    # Special case: Plane is 3D only
    if (type0 is Plane or type1 is Plane) and obj0.dims != 3:
        raise ValueError(f"{func_prefix} with Plane is only supported in 3D")

    # Build suffix: primitives have no index/ngon
    meta = InputMeta(None, obj0.dtype, None, obj0.dims)
    suffix = build_suffix(meta)
    func_name = func_template.format(func_prefix, suffix)
    cpp_func = getattr(_trueform.core, func_name)

    # Handle symmetry
    if needs_swap:
        return cpp_func(obj1.data, obj0.data)
    return cpp_func(obj0.data, obj1.data)


def _spatial_distance(obj0, obj1, is_form0: bool, is_form1: bool, func_prefix: str) -> float:
    """Form x primitive or form x form distance via neighbor_search."""
    from ._spatial import neighbor_search

    # Ensure form is first argument (neighbor_search expects this)
    if is_form0:
        result = neighbor_search(obj0, obj1, radius=None)
    else:
        result = neighbor_search(obj1, obj0, radius=None)

    # Extract metric based on form combination
    if is_form0 and is_form1:
        # Form-form: ((idx0, idx1), (distance_squared, pt0, pt1))
        metric = result[1][0]
    else:
        # Form-primitive: (index, distance_squared, point)
        metric = result[1]

    # Return based on function type
    return np.sqrt(metric) if func_prefix == "distance" else metric
