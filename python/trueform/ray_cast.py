"""
Unified ray casting API

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via ziga.sajovic@xlab.si.
https://github.com/xlabmedical/trueform
"""

import numpy as np
from typing import Any, Optional
from . import _trueform
from .core._ray_cast import _RAY_CAST_DISPATCH as _CORE_DISPATCH
from .primitives import Plane


# Unified dispatch table
# Merges core (primitives) and spatial (meshes, point clouds, etc.) dispatch tables
_RAY_CAST_DISPATCH = {**_CORE_DISPATCH}
# When spatial module is added:
# from .spatial._ray_cast import _RAY_CAST_DISPATCH as _SPATIAL_DISPATCH
# _RAY_CAST_DISPATCH = {**_CORE_DISPATCH, **_SPATIAL_DISPATCH}


def ray_cast(ray: Any, target: Any) -> Optional[float]:
    """
    Cast a ray against a geometric object and return the parametric distance t if it hits.

    Returns None if no intersection, or the parametric distance t where:
        hit_point = ray.origin + t * ray.direction

    This function works with both core primitives (Point, Segment, Polygon, Line, AABB, Plane)
    and spatial data structures (Mesh, PointCloud - when spatial module is available).

    Parameters
    ----------
    ray : Ray
        The ray to cast
    target : Segment, Polygon, Line, AABB, Plane, Mesh, PointCloud, etc.
        The geometric object to test against

    Returns
    -------
    t : float or None
        Parametric distance along the ray if intersection occurs, None otherwise

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> # Ray pointing down at a triangle in xy-plane
    >>> ray = tf.Ray(origin=[0.5, 0.3, 2.0], direction=[0.0, 0.0, -1.0])
    >>> triangle = tf.Polygon([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
    >>> t = tf.ray_cast(ray, triangle)
    >>> if t is not None:
    ...     hit_point = ray.origin + t * ray.direction
    ...     print(f"Hit at {hit_point}, t={t}")
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

    # Check if target type is supported
    target_type = type(target)
    if target_type not in _RAY_CAST_DISPATCH:
        supported = ", ".join(t.__name__ for t in _RAY_CAST_DISPATCH.keys())
        raise TypeError(
            f"ray_cast not implemented for target type: {target_type.__name__}. "
            f"Supported types: {supported}"
        )

    # Special case: Plane is 3D only
    if target_type is Plane and ray.dims != 3:
        raise ValueError("ray_cast with Plane is only supported in 3D")

    # Get variant suffix (e.g., "float3d" or "double2d")
    dtype_str = 'float' if ray.dtype == np.float32 else 'double'
    suffix = f"{dtype_str}{ray.dims}d"

    # Dispatch to appropriate C++ function
    func_name = _RAY_CAST_DISPATCH[target_type].format(suffix)
    return getattr(_trueform, func_name)(ray.data, target.data)
