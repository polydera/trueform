"""
Unified intersects API

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Any
from .. import _trueform
from .._primitives.primitive import Primitive
from .._dispatch import InputMeta, extract_meta, build_suffix, build_suffix_pair, canonicalize_index_order
from ._dispatch import INTERSECTS_FORM_FORM


def intersects(obj0: Any, obj1: Any):
    """
    Check whether two geometric objects intersect.

    This function works with both core primitives (Point, Segment, Triangle, Polygon,
    Line, AABB, Ray, Plane) and spatial data structures (Mesh, EdgeMesh, PointCloud).
    Supports batch broadcasting: single x single returns a bool, batch operations
    return an ndarray.

    Parameters
    ----------
    obj0, obj1 : geometric objects
        Any combination of Point, Segment, Triangle, Polygon, Line, AABB, Ray, Plane,
        Mesh, EdgeMesh, PointCloud

    Returns
    -------
    bool or np.ndarray
        True if the objects intersect, False otherwise.
        For batch operations, returns ndarray of int8 (0/1).

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

    >>> # Check if point intersects mesh
    >>> faces = np.array([[0, 1, 2]], dtype=np.int32)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>> pt = tf.Point([0.5, 0.3, 0.0])
    >>> tf.intersects(mesh, pt)
    True
    """
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
        fn = getattr(_trueform.spatial, f"intersects_pp_{suffix}")
        return fn(obj0._wrapper, obj1._wrapper)

    if is_form0 and is_prim1:
        meta = extract_meta(obj0)
        suffix = build_suffix(meta)
        fn = getattr(_trueform.spatial, f"intersects_{meta.form_name}_fp_{suffix}")
        return fn(obj0._wrapper, obj1._wrapper)

    if is_prim0 and is_form1:
        meta = extract_meta(obj1)
        suffix = build_suffix(meta)
        fn = getattr(_trueform.spatial, f"intersects_{meta.form_name}_fp_{suffix}")
        return fn(obj1._wrapper, obj0._wrapper)

    if is_form0 and is_form1:
        type_pair = (type(obj0), type(obj1))
        if type_pair not in INTERSECTS_FORM_FORM:
            raise TypeError(
                f"intersects not supported for types: {type(obj0).__name__}, {type(obj1).__name__}")
        func_template, needs_swap = INTERSECTS_FORM_FORM[type_pair]
        form0 = obj1 if needs_swap else obj0
        form1 = obj0 if needs_swap else obj1
        form0, form1, _ = canonicalize_index_order(form0, form1)
        meta0 = extract_meta(form0)
        meta1 = extract_meta(form1)
        suffix = build_suffix_pair(meta0, meta1)
        fn = getattr(_trueform.spatial, func_template.format(suffix))
        return fn(form0._wrapper, form1._wrapper)

    raise TypeError(
        f"intersects not supported for types: {type(obj0).__name__}, {type(obj1).__name__}")
