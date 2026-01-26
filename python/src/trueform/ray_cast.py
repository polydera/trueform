"""
Unified ray casting API

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Any, Optional, Tuple
from . import _trueform
from ._primitives import Plane

# Dispatch infrastructure
from ._dispatch import InputMeta, extract_meta, build_suffix
from ._core._dispatch import RAY_CAST as CORE_RAY_CAST
from ._spatial._dispatch import RAY_CAST as SPATIAL_RAY_CAST


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
        raise TypeError("Both ray and target must have 'dims' attribute")

    if ray.dims != target.dims:
        raise ValueError(
            f"Dimension mismatch: ray has {ray.dims}D, target has {target.dims}D. "
            f"Both objects must have the same dimensionality (2D or 3D)."
        )

    # Validate dtypes match
    if not hasattr(ray, 'dtype') or not hasattr(target, 'dtype'):
        raise TypeError("Both ray and target must have 'dtype' attribute")

    if ray.dtype != target.dtype:
        raise TypeError(
            f"Dtype mismatch: ray has {ray.dtype}, target has {target.dtype}. "
            f"Both objects must have the same dtype (float32 or float64)."
        )

    target_type = type(target)

    if target_type in SPATIAL_RAY_CAST:
        return _spatial_ray_cast(ray, target, target_type, config)
    elif target_type in CORE_RAY_CAST:
        return _core_ray_cast(ray, target, target_type, config)
    else:
        supported_core = ", ".join(t.__name__ for t in CORE_RAY_CAST.keys())
        supported_spatial = ", ".join(t.__name__ for t in SPATIAL_RAY_CAST.keys())
        raise TypeError(
            f"ray_cast not implemented for target type: {target_type.__name__}. "
            f"Supported types: {supported_core}, {supported_spatial}"
        )


def _core_ray_cast(ray, target, target_type, config):
    """Ray cast against core primitives."""
    # Special case: Plane is 3D only
    if target_type is Plane and ray.dims != 3:
        raise ValueError("ray_cast with Plane is only supported in 3D")

    # Build suffix: primitives have no index/ngon, just real+dims
    meta = InputMeta(None, ray.dtype, None, ray.dims)
    suffix = build_suffix(meta)
    func_template = CORE_RAY_CAST[target_type]
    func_name = func_template.format(suffix)
    cpp_func = getattr(_trueform.core, func_name)

    return cpp_func(ray.data, target.data, config)


def _spatial_ray_cast(ray, target, target_type, config):
    """Ray cast against spatial forms."""
    # Build suffix using extract_meta + build_suffix
    meta = extract_meta(target)
    suffix = build_suffix(meta)
    func_template = SPATIAL_RAY_CAST[target_type]
    func_name = func_template.format(suffix)
    cpp_func = getattr(_trueform.spatial, func_name)

    return cpp_func(ray.data, target._wrapper, config)
