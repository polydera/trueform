"""
Unified intersects API

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Any
from . import _trueform
from .core._intersects import _INTERSECTS_DISPATCH as _CORE_DISPATCH
from .primitives import Plane


# Unified dispatch table
# Merges core (primitives) and spatial (meshes, point clouds, etc.) dispatch tables
_INTERSECTS_DISPATCH = {**_CORE_DISPATCH}
# When spatial module is added:
# from .spatial._intersects import _INTERSECTS_DISPATCH as _SPATIAL_DISPATCH
# _INTERSECTS_DISPATCH = {**_CORE_DISPATCH, **_SPATIAL_DISPATCH}


def intersects(obj0: Any, obj1: Any) -> bool:
    """
    Check whether two geometric objects intersect.

    This function works with both core primitives (Point, Segment, Polygon, Line, AABB, Ray, Plane)
    and spatial data structures (Mesh, PointCloud - when spatial module is available).

    Parameters
    ----------
    obj0, obj1 : geometric objects
        Any combination of Point, Segment, Polygon, Line, AABB, Ray, Plane, Mesh, PointCloud, etc.

    Returns
    -------
    bool
        True if the objects intersect, False otherwise

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> # Check if point is inside AABB
    >>> pt = tf.Point([0.5, 0.5])
    >>> box = tf.AABB(min=[0, 0], max=[1, 1])
    >>> tf.intersects(pt, box)
    True

    >>> # Check if two segments intersect
    >>> seg1 = tf.Segment([[0, 0], [1, 1]])
    >>> seg2 = tf.Segment([[0, 1], [1, 0]])
    >>> tf.intersects(seg1, seg2)
    True
    """

    # Helper to get dimensionality
    def get_dims(obj):
        if hasattr(obj, 'dims'):
            return obj.dims
        raise TypeError(f"Cannot determine dimensions for type {type(obj)}")

    # Helper to get variant suffix
    def get_suffix(obj):
        if hasattr(obj, 'dtype') and hasattr(obj, 'dims'):
            dtype_str = 'float' if obj.dtype == np.float32 else 'double'
            return f"{dtype_str}{obj.dims}d"
        raise TypeError(f"Cannot determine variant for type {type(obj)}")

    # Helper to extract data from object
    def get_data(obj):
        if hasattr(obj, 'data'):
            return obj.data
        raise TypeError(f"Cannot extract data from type {type(obj)}")

    # Validate dimensions match
    dims0 = get_dims(obj0)
    dims1 = get_dims(obj1)
    if dims0 != dims1:
        raise ValueError(
            f"Dimension mismatch: obj0 has {dims0}D, obj1 has {dims1}D. "
            f"Both objects must have the same dimensionality (2D or 3D)."
        )

    # Normalize types for dispatch
    type0 = type(obj0)
    type1 = type(obj1)

    # Look up dispatch info
    type_pair = (type0, type1)
    if type_pair not in _INTERSECTS_DISPATCH:
        supported = set()
        for t0, t1 in _INTERSECTS_DISPATCH.keys():
            supported.add(t0.__name__)
            supported.add(t1.__name__)
        raise TypeError(
            f"intersects not implemented for types: {type0.__name__}, {type1.__name__}. "
            f"Supported types: {', '.join(sorted(supported))}"
        )

    func_template, needs_swap = _INTERSECTS_DISPATCH[type_pair]

    # Special case: Plane is 3D only
    if (type0 is Plane or type1 is Plane) and dims0 != 3:
        raise ValueError("intersects with Plane is only supported in 3D")

    # Get suffix and function name
    suffix = get_suffix(obj0 if not needs_swap else obj1)
    func_name = func_template.format(suffix)

    # Get data and call C++ function
    data0 = get_data(obj0)
    data1 = get_data(obj1)

    if needs_swap:
        return getattr(_trueform, func_name)(data1, data0)
    else:
        return getattr(_trueform, func_name)(data0, data1)
