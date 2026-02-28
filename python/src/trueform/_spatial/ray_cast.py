"""
Unified ray casting API

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Any, Optional, Tuple, Union
import numpy as np
from .. import _trueform
from .._primitives.primitive import Primitive
from .._dispatch import InputMeta, extract_meta, build_suffix


def _prepare_config(config, dtype, n):
    """Expand config to (ndarray, ndarray) for C++ binding.

    n == 0: single call → None stays None, scalar tuple → 1-element arrays.
    n > 0:  batch call → scalar tuple or None → full arrays of length n.
    Per-ray arrays pass through unchanged.
    """
    if config is None:
        if n == 0:
            return None
        return (np.zeros(n, dtype=dtype),
                np.full(n, np.finfo(dtype).max, dtype=dtype))

    min_v, max_v = config
    if isinstance(min_v, np.ndarray):
        return config  # already per-ray arrays

    if n == 0:
        return (np.array([min_v], dtype=dtype),
                np.array([max_v], dtype=dtype))

    return (np.full(n, min_v, dtype=dtype),
            np.full(n, max_v, dtype=dtype))


def ray_cast(ray: Any, target: Any, config=None):
    """
    Cast a ray against a geometric object and return intersection information.

    For core primitives (Segment, Triangle, Polygon, Line, AABB, Plane):
        Returns the parametric distance t if intersection occurs, None otherwise.
        hit_point = ray.origin + t * ray.direction

    For spatial structures (PointCloud, Mesh, EdgeMesh):
        Returns (element_index, t) if intersection occurs, None otherwise.
        - element_index: index of the intersected element (point/face/edge)
        - t: parametric distance along the ray

    Supports batch rays and/or batch targets. For batch operations against
    primitives, returns ndarray of t values (NaN for misses). For batch
    operations against forms, returns (element_ids, t_values) — matching
    the single-ray return order.

    Parameters
    ----------
    ray : Ray
        The ray to cast (single or batch)
    target : Segment, Triangle, Polygon, Line, AABB, Plane, Mesh, EdgeMesh, or PointCloud
        The geometric object to test against
    config : tuple[float, float] or tuple[ndarray, ndarray] or None, optional
        Ray configuration constraining the parametric range.
        - ``(min_t, max_t)``: single range applied to all rays
        - ``(min_ts, max_ts)``: per-ray arrays of shape ``(N,)``, same dtype as ray
        If None, uses default (0.0 to infinity).

    Returns
    -------
    result : float, tuple, np.ndarray, or None
        - For core primitives: float (parametric distance t) or None
        - For spatial structures: tuple (element_index, t) or None
        - For batch rays × primitive: ndarray of t values (NaN for misses)
        - For batch rays × form: (ndarray of element IDs, ndarray of t values)
          with -1/NaN for misses

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> # Ray casting against a triangle
    >>> ray = tf.Ray(origin=[0.5, 0.3, 2.0], direction=[0.0, 0.0, -1.0])
    >>> triangle = tf.Triangle(a=[0, 0, 0], b=[1, 0, 0], c=[0.5, 1, 0])
    >>> t = tf.ray_cast(ray, triangle)
    >>> if t is not None:
    ...     hit_point = ray.origin + t * ray.direction
    ...     print(f"Hit at {hit_point}, t={t}")
    >>>
    >>> # Ray casting against a mesh with custom range
    >>> mesh = tf.Mesh(faces, points)
    >>> ray = tf.Ray(origin=[0.3, 0.3, 2.0], direction=[0.0, 0.0, -1.0])
    >>> result = tf.ray_cast(ray, mesh, config=(0.5, 10.0))
    >>>
    >>> # Per-ray config for batch rays
    >>> rays = tf.Ray(origin=origins, direction=directions)
    >>> min_ts = np.zeros(n, dtype=np.float32)
    >>> max_ts = np.full(n, 10.0, dtype=np.float32)
    >>> ids, ts = tf.ray_cast(rays, mesh, config=(min_ts, max_ts))
    """
    from . import Mesh, EdgeMesh, PointCloud

    is_prim = isinstance(target, Primitive)
    is_form = isinstance(target, (Mesh, EdgeMesh, PointCloud))

    if ray.dims != target.dims:
        raise ValueError(
            f"Dimension mismatch: ray has {ray.dims}D, target has {target.dims}D. "
            f"Both objects must have the same dimensionality (2D or 3D).")

    if ray.dtype != target.dtype:
        raise TypeError(
            f"Dtype mismatch: ray has {ray.dtype}, target has {target.dtype}. "
            f"Both objects must have the same dtype.")

    if is_prim:
        # Batch when either ray or target (or both) is batch
        br, bt = ray.is_batch, target.is_batch
        n = ray.count if br else (target.count if bt else 0)
        config = _prepare_config(config, ray.dtype, n)
        suffix = build_suffix(InputMeta(None, ray.dtype, None, ray.dims))
        fn = getattr(_trueform.spatial, f"ray_cast_pp_{suffix}")
        return fn(ray._wrapper, target._wrapper, config)

    if is_form:
        # Batch only when ray is batch (forms are not batch primitives)
        n = ray.count if ray.is_batch else 0
        config = _prepare_config(config, ray.dtype, n)
        meta = extract_meta(target)
        suffix = build_suffix(meta)
        fn = getattr(_trueform.spatial, f"ray_cast_{meta.form_name}_fp_{suffix}")
        return fn(ray._wrapper, target._wrapper, config)

    raise TypeError(
        f"ray_cast not supported for target type: {type(target).__name__}")
