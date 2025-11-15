"""
Unified ray casting API

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Any, Optional, Tuple
from . import _trueform
from ._core._ray_cast import _RAY_CAST_DISPATCH as _CORE_DISPATCH
from ._spatial._ray_cast import _SPATIAL_RAY_CAST_DISPATCH
from ._primitives import Plane


# Core primitives dispatch table
_RAY_CAST_DISPATCH = {**_CORE_DISPATCH}


def ray_cast(ray: Any, target: Any, config: Optional[Tuple[float, float]] = None):
    """
    Cast a ray against a geometric object and return intersection information.

    For core primitives (Segment, Polygon, Line, AABB, Plane):
        Returns the parametric distance t if intersection occurs, None otherwise.
        hit_point = ray.origin + t * ray.direction

    For spatial structures (PointCloud, Mesh, EdgeMesh):
        Returns (element_index, t) if intersection occurs, None otherwise.
        - element_index: index of the intersected element (point/face/edge)
        - t: parametric distance along the ray

    Parameters
    ----------
    ray : Ray
        The ray to cast
    target : Segment, Polygon, Line, AABB, Plane, Mesh, EdgeMesh, or PointCloud
        The geometric object to test against
    config : tuple[float, float] or None, optional
        Ray configuration (min_t, max_t) to constrain the ray casting range.
        - min_t: minimum parametric distance (default: 0.0)
        - max_t: maximum parametric distance (default: infinity)
        Both float('inf') and np.inf are supported for unbounded ranges.
        If None, uses default configuration.

    Returns
    -------
    result : float, tuple[int, float], or None
        - For core primitives: float (parametric distance t) or None
        - For spatial structures: tuple (element_index, t) or None

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> # Ray casting against a polygon
    >>> ray = tf.Ray(origin=[0.5, 0.3, 2.0], direction=[0.0, 0.0, -1.0])
    >>> triangle = tf.Polygon([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
    >>> t = tf.ray_cast(ray, triangle)
    >>> if t is not None:
    ...     hit_point = ray.origin + t * ray.direction
    ...     print(f"Hit at {hit_point}, t={t}")
    >>>
    >>> # Ray casting against a mesh with custom range
    >>> faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    >>> points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0, 1]], dtype=np.float32)
    >>> mesh = tf.Mesh(faces, points)
    >>> ray = tf.Ray(origin=[0.3, 0.3, 2.0], direction=[0.0, 0.0, -1.0])
    >>> # Only check intersections between t=0.5 and t=10.0
    >>> result = tf.ray_cast(ray, mesh, config=(0.5, 10.0))
    >>> if result is not None:
    ...     face_idx, t = result
    ...     print(f"Hit face {face_idx} at t={t}")
    >>>\
    >>> # Using np.inf for unbounded range (equivalent to default)
    >>> result = tf.ray_cast(ray, mesh, config=(0.0, np.inf))
    """

    # Validate dimensions match
    if not hasattr(ray, 'dims') or not hasattr(target, 'dims'):
        raise TypeError(f"Both ray and target must have 'dims' attribute")

    if ray.dims != target.dims:
        raise ValueError(
            f"Dimension mismatch: ray has {ray.dims}D, target has {target.dims}D. "
            f"Both objects must have the same dimensionality (2D or 3D)."
        )

    # Validate dtypes match
    if not hasattr(ray, 'dtype') or not hasattr(target, 'dtype'):
        raise TypeError(f"Both ray and target must have 'dtype' attribute")

    if ray.dtype != target.dtype:
        raise TypeError(
            f"Dtype mismatch: ray has {ray.dtype}, target has {target.dtype}. "
            f"Both objects must have the same dtype (float32 or float64)."
        )

    target_type = type(target)
    target_type_name = target_type.__name__

    # Handle spatial structures (PointCloud, Mesh, EdgeMesh)
    if target_type_name in _SPATIAL_RAY_CAST_DISPATCH:
        # Compute appropriate suffix based on object type
        if target_type_name == 'PointCloud':
            # PointCloud: suffix is "float2d" or "double3d"
            dtype_str = 'float' if target.points.dtype == np.float32 else 'double'
            suffix = f"{dtype_str}{target.dims}d"
        elif target_type_name == 'Mesh':
            # Mesh: suffix is "intfloat32d" or "int64double43d"
            faces_dtype = target.faces.dtype
            points_dtype = target.points.dtype
            ngon = target.ngon

            index_str = 'int' if faces_dtype == np.int32 else 'int64'
            real_str = 'float' if points_dtype == np.float32 else 'double'
            suffix = f"{index_str}{real_str}{ngon}{target.dims}d"
        else:  # EdgeMesh
            # EdgeMesh: suffix is "intfloat2d" or "int64double3d"
            edges_dtype = target.edges.dtype
            points_dtype = target.points.dtype

            index_str = 'int' if edges_dtype == np.int32 else 'int64'
            real_str = 'float' if points_dtype == np.float32 else 'double'
            suffix = f"{index_str}{real_str}{target.dims}d"

        # Get function name from dispatch table and call C++ function
        func_name = _SPATIAL_RAY_CAST_DISPATCH[target_type_name].format(suffix)
        cpp_func = getattr(_trueform.spatial, func_name)
        return cpp_func(ray.data, target._wrapper, config)

    # Handle core primitives (Segment, Polygon, Line, AABB, Plane)
    if target_type not in _RAY_CAST_DISPATCH:
        supported_core = ", ".join(t.__name__ for t in _RAY_CAST_DISPATCH.keys())
        supported_spatial = ", ".join(_SPATIAL_RAY_CAST_DISPATCH.keys())
        raise TypeError(
            f"ray_cast not implemented for target type: {target_type_name}. "
            f"Supported types: {supported_core}, {supported_spatial}"
        )

    # Special case: Plane is 3D only
    if target_type is Plane and ray.dims != 3:
        raise ValueError("ray_cast with Plane is only supported in 3D")

    # Get variant suffix (e.g., "float3d" or "double2d")
    dtype_str = 'float' if ray.dtype == np.float32 else 'double'
    suffix = f"{dtype_str}{ray.dims}d"

    # Dispatch to appropriate C++ function for core primitives
    func_name = _RAY_CAST_DISPATCH[target_type].format(suffix)
    return getattr(_trueform, func_name)(ray.data, target.data, config)
