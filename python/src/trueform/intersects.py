"""
Unified intersects API

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""
from typing import Any
from . import _trueform
from ._primitives import Plane

# Dispatch infrastructure
from ._dispatch import (
    InputMeta,
    extract_meta,
    build_suffix,
    build_suffix_pair,
    canonicalize_index_order,
)
from ._core._dispatch import INTERSECTS as CORE_INTERSECTS
from ._spatial._dispatch import INTERSECTS_FORM_PRIM, INTERSECTS_FORM_FORM


def intersects(obj0: Any, obj1: Any) -> bool:
    """
    Check whether two geometric objects intersect.

    This function works with both core primitives (Point, Segment, Polygon, Line, AABB, Ray, Plane)
    and spatial data structures (Mesh, EdgeMesh, PointCloud).

    Parameters
    ----------
    obj0, obj1 : geometric objects
        Any combination of Point, Segment, Polygon, Line, AABB, Ray, Plane, Mesh, EdgeMesh, PointCloud

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

    >>> # Check if ray intersects mesh
    >>> faces = np.array([[0, 1, 2]], dtype=np.int32)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>> pt = tf.Point([0.5, 0.3, 0.0])
    >>> tf.intersects(mesh, pt)
    True
    """
    # Validate dimensions match
    if not hasattr(obj0, 'dims') or not hasattr(obj1, 'dims'):
        raise TypeError("Both objects must have 'dims' attribute")

    if obj0.dims != obj1.dims:
        raise ValueError(
            f"Dimension mismatch: obj0 has {obj0.dims}D, obj1 has {obj1.dims}D. "
            f"Both objects must have the same dimensionality (2D or 3D)."
        )

    # Get types
    type0 = type(obj0)
    type1 = type(obj1)
    type_pair = (type0, type1)

    # Import spatial form types
    from ._spatial import Mesh, EdgeMesh, PointCloud

    # Check if this is a spatial operation
    is_spatial = type0 in (Mesh, EdgeMesh, PointCloud) or type1 in (Mesh, EdgeMesh, PointCloud)

    if is_spatial:
        return _spatial_intersects(obj0, obj1, type0, type1, type_pair)
    else:
        return _core_intersects(obj0, obj1, type0, type1, type_pair)


def _core_intersects(obj0, obj1, type0, type1, type_pair):
    """Primitive x primitive intersects."""
    if type_pair not in CORE_INTERSECTS:
        supported = {t.__name__ for pair in CORE_INTERSECTS.keys() for t in pair}
        raise TypeError(
            f"intersects not implemented for types: {type0.__name__}, {type1.__name__}. "
            f"Supported core types: {', '.join(sorted(supported))}"
        )

    func_template, needs_swap = CORE_INTERSECTS[type_pair]

    # Special case: Plane is 3D only
    if (type0 is Plane or type1 is Plane) and obj0.dims != 3:
        raise ValueError("intersects with Plane is only supported in 3D")

    # Build suffix: primitives have no index/ngon
    meta = InputMeta(None, obj0.dtype, None, obj0.dims)
    suffix = build_suffix(meta)
    func_name = func_template.format(suffix)
    cpp_func = getattr(_trueform.core, func_name)

    # Handle symmetry
    if needs_swap:
        return cpp_func(obj1.data, obj0.data)
    return cpp_func(obj0.data, obj1.data)


def _spatial_intersects(obj0, obj1, type0, type1, type_pair):
    """Form x primitive or form x form intersects."""
    from ._spatial import Mesh, EdgeMesh, PointCloud

    both_forms = (type0 in (Mesh, EdgeMesh, PointCloud) and
                  type1 in (Mesh, EdgeMesh, PointCloud))

    if both_forms:
        return _form_form_intersects(obj0, obj1, type0, type1, type_pair)
    else:
        return _form_prim_intersects(obj0, obj1, type0, type1, type_pair)


def _form_prim_intersects(obj0, obj1, type0, type1, type_pair):
    """Form x primitive intersects."""

    if type_pair not in INTERSECTS_FORM_PRIM:
        supported = {t.__name__ for pair in INTERSECTS_FORM_PRIM.keys() for t in pair}
        raise TypeError(
            f"intersects not implemented for types: {type0.__name__}, {type1.__name__}. "
            f"Supported spatial types: {', '.join(sorted(supported))}"
        )

    func_template, needs_swap = INTERSECTS_FORM_PRIM[type_pair]

    # Determine form and primitive based on swap
    form_obj = obj1 if needs_swap else obj0
    prim_obj = obj0 if needs_swap else obj1

    # Build suffix using extract_meta + build_suffix
    meta = extract_meta(form_obj)
    suffix = build_suffix(meta)
    func_name = func_template.format(suffix)
    cpp_func = getattr(_trueform.spatial, func_name)

    return cpp_func(form_obj._wrapper, prim_obj.data)


def _form_form_intersects(obj0, obj1, type0, type1, type_pair):
    """Form x form intersects."""

    if type_pair not in INTERSECTS_FORM_FORM:
        supported = {t.__name__ for pair in INTERSECTS_FORM_FORM.keys() for t in pair}
        raise TypeError(
            f"intersects not implemented for types: {type0.__name__}, {type1.__name__}. "
            f"Supported form-form types: {', '.join(sorted(supported))}"
        )

    func_template, needs_swap = INTERSECTS_FORM_FORM[type_pair]

    # Apply dispatch table swap
    form0_obj = obj1 if needs_swap else obj0
    form1_obj = obj0 if needs_swap else obj1

    # Apply index canonicalization (int32 before int64 for same types)
    form0_obj, form1_obj, _ = canonicalize_index_order(form0_obj, form1_obj)

    # Build suffix using extract_meta + build_suffix_pair
    meta0 = extract_meta(form0_obj)
    meta1 = extract_meta(form1_obj)
    suffix = build_suffix_pair(meta0, meta1)

    func_name = func_template.format(suffix)
    cpp_func = getattr(_trueform.spatial, func_name)

    return cpp_func(form0_obj._wrapper, form1_obj._wrapper)
