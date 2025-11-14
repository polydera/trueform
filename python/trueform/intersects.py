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
from ._core._intersects import _INTERSECTS_DISPATCH as _CORE_DISPATCH
from ._primitives import Plane


# Import spatial dispatch table (kept separate from core)
from ._spatial._intersects import _INTERSECTS_DISPATCH as _SPATIAL_INTERSECTS_DISPATCH


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
        raise TypeError(f"Both objects must have 'dims' attribute")

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
    from .core.mesh import Mesh
    from .core.edge_mesh import EdgeMesh
    from .core.point_cloud import PointCloud

    # Check if this is a spatial (form-primitive) operation
    is_spatial = type0 in (Mesh, EdgeMesh, PointCloud) or type1 in (Mesh, EdgeMesh, PointCloud)

    if is_spatial:
        # Handle spatial structures (Mesh, EdgeMesh, PointCloud)
        if type_pair not in _SPATIAL_INTERSECTS_DISPATCH:
            supported_spatial = set()
            for t0, t1 in _SPATIAL_INTERSECTS_DISPATCH.keys():
                supported_spatial.add(t0.__name__)
                supported_spatial.add(t1.__name__)
            raise TypeError(
                f"intersects not implemented for types: {type0.__name__}, {type1.__name__}. "
                f"Supported spatial types: {', '.join(sorted(supported_spatial))}"
            )

        func_template, needs_swap = _SPATIAL_INTERSECTS_DISPATCH[type_pair]

        # Check if this is form-form or form-primitive
        both_forms = (type0 in (Mesh, EdgeMesh, PointCloud) and
                      type1 in (Mesh, EdgeMesh, PointCloud))

        if both_forms:
            # Form-Form intersection
            form0_obj = obj0 if not needs_swap else obj1
            form1_obj = obj1 if not needs_swap else obj0

            # Compute suffix based on both forms
            # For form-form, suffix encodes both forms' properties
            form0_type = type(form0_obj)
            form1_type = type(form1_obj)

            # Canonicalize index type ordering for same-type forms
            # C++ only implements: int×int, int×int64, int64×int64
            # If we have int64×int, swap to int×int64
            extra_swap = False
            if form0_type == form1_type:
                if form0_type is EdgeMesh:
                    index0_dtype = form0_obj.edges.dtype
                    index1_dtype = form1_obj.edges.dtype
                    # Swap if form0 is int64 and form1 is int32
                    if index0_dtype == np.int64 and index1_dtype == np.int32:
                        form0_obj, form1_obj = form1_obj, form0_obj
                        extra_swap = True
                elif form0_type is Mesh:
                    index0_dtype = form0_obj.faces.dtype
                    index1_dtype = form1_obj.faces.dtype
                    # Swap if form0 is int64 and form1 is int32
                    if index0_dtype == np.int64 and index1_dtype == np.int32:
                        form0_obj, form1_obj = form1_obj, form0_obj
                        extra_swap = True

            # Get real type (must match)
            if form0_type is PointCloud:
                real_str = 'float' if form0_obj.points.dtype == np.float32 else 'double'
            elif form0_type is Mesh:
                real_str = 'float' if form0_obj.points.dtype == np.float32 else 'double'
            else:  # EdgeMesh
                real_str = 'float' if form0_obj.points.dtype == np.float32 else 'double'

            # Dims (must match)
            dims_str = f"{form0_obj.dims}d"

            # Build suffix based on form types
            if form0_type is PointCloud and form1_type is PointCloud:
                suffix = f"{real_str}{dims_str}"
            elif form0_type is EdgeMesh and form1_type is EdgeMesh:
                index0_str = 'int' if form0_obj.edges.dtype == np.int32 else 'int64'
                index1_str = 'int' if form1_obj.edges.dtype == np.int32 else 'int64'
                suffix = f"{index0_str}{index1_str}{real_str}{dims_str}"
            elif form0_type is EdgeMesh and form1_type is PointCloud:
                index0_str = 'int' if form0_obj.edges.dtype == np.int32 else 'int64'
                suffix = f"{index0_str}{real_str}{dims_str}"
            elif form0_type is Mesh and form1_type is PointCloud:
                index0_str = 'int' if form0_obj.faces.dtype == np.int32 else 'int64'
                suffix = f"{index0_str}{real_str}{form0_obj.ngon}{dims_str}"
            elif form0_type is Mesh and form1_type is EdgeMesh:
                index0_str = 'int' if form0_obj.faces.dtype == np.int32 else 'int64'
                index1_str = 'int' if form1_obj.edges.dtype == np.int32 else 'int64'
                suffix = f"{index0_str}{index1_str}{real_str}{form0_obj.ngon}{dims_str}"
            elif form0_type is Mesh and form1_type is Mesh:
                index0_str = 'int' if form0_obj.faces.dtype == np.int32 else 'int64'
                index1_str = 'int' if form1_obj.faces.dtype == np.int32 else 'int64'
                suffix = f"{index0_str}{index1_str}{form0_obj.ngon}{form1_obj.ngon}{real_str}{dims_str}"
            else:
                raise TypeError(f"Unexpected form-form combination: {form0_type}, {form1_type}")

            # Get function and call
            func_name = func_template.format(suffix)
            cpp_func = getattr(_trueform.spatial, func_name)

            if needs_swap:
                return cpp_func(form0_obj._wrapper, form1_obj._wrapper)
            else:
                return cpp_func(form0_obj._wrapper, form1_obj._wrapper)
        else:
            # Form-Primitive intersection
            form_obj = obj0 if not needs_swap else obj1
            prim_obj = obj1 if not needs_swap else obj0

            # Compute suffix based on form type
            form_type = type(form_obj)
            if form_type is PointCloud:
                dtype_str = 'float' if form_obj.points.dtype == np.float32 else 'double'
                suffix = f"{dtype_str}{form_obj.dims}d"
            elif form_type is Mesh:
                faces_dtype = form_obj.faces.dtype
                points_dtype = form_obj.points.dtype
                index_str = 'int' if faces_dtype == np.int32 else 'int64'
                real_str = 'float' if points_dtype == np.float32 else 'double'
                suffix = f"{index_str}{real_str}{form_obj.ngon}{form_obj.dims}d"
            elif form_type is EdgeMesh:
                edges_dtype = form_obj.edges.dtype
                points_dtype = form_obj.points.dtype
                index_str = 'int' if edges_dtype == np.int32 else 'int64'
                real_str = 'float' if points_dtype == np.float32 else 'double'
                suffix = f"{index_str}{real_str}{form_obj.dims}d"
            else:
                raise TypeError(f"Unexpected form type: {form_type}")

            # Get function and call
            func_name = func_template.format(suffix)
            cpp_func = getattr(_trueform.spatial, func_name)

            if needs_swap:
                return cpp_func(form_obj._wrapper, prim_obj.data)
            else:
                return cpp_func(form_obj._wrapper, prim_obj.data)

    # Handle core primitives (primitive-primitive)
    if type_pair not in _CORE_DISPATCH:
        supported_core = set()
        for t0, t1 in _CORE_DISPATCH.keys():
            supported_core.add(t0.__name__)
            supported_core.add(t1.__name__)
        raise TypeError(
            f"intersects not implemented for types: {type0.__name__}, {type1.__name__}. "
            f"Supported core types: {', '.join(sorted(supported_core))}"
        )

    func_template, needs_swap = _CORE_DISPATCH[type_pair]

    # Special case: Plane is 3D only
    if (type0 is Plane or type1 is Plane) and obj0.dims != 3:
        raise ValueError("intersects with Plane is only supported in 3D")

    # Compute suffix (both primitives have same dtype)
    dtype_str = 'float' if obj0.dtype == np.float32 else 'double'
    suffix = f"{dtype_str}{obj0.dims}d"

    # Get function and call
    func_name = func_template.format(suffix)
    cpp_func = getattr(_trueform, func_name)

    if needs_swap:
        return cpp_func(obj1.data, obj0.data)
    else:
        return cpp_func(obj0.data, obj1.data)
