"""
Vectorized distance field computation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Any
from . import _trueform
from ._primitives import Plane, Segment, Polygon, Line, AABB
from ._dispatch import extract_meta, build_suffix


# Dispatch table for primitives
_DISTANCE_FIELD_DISPATCH = {
    Plane: "distance_field_plane_{}",
    Segment: "distance_field_segment_{}",
    Polygon: "distance_field_polygon_{}",
    Line: "distance_field_line_{}",
    AABB: "distance_field_aabb_{}",
}


def distance_field(points: Any, primitive: Any) -> np.ndarray:
    """
    Compute distance from points to a primitive (vectorized, parallel).

    This function efficiently computes distances from many points to a geometric
    primitive using parallel computation. For Plane primitives, returns signed
    distance (negative inside, positive outside). For other primitives, returns
    unsigned (Euclidean) distance.

    Parameters
    ----------
    points : np.ndarray or PointCloud
        Points to compute distances for:
        - numpy array with shape (N, 2) or (N, 3)
        - or PointCloud object
    primitive : Plane, Segment, Polygon, Line, or AABB
        Target geometric primitive

    Returns
    -------
    np.ndarray
        Distances with shape (N,)
        - Signed distance for Plane (negative inside, positive outside)
        - Unsigned Euclidean distance for other primitives

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> # Distance to a plane
    >>> points = np.random.rand(1000, 3).astype(np.float32)
    >>> plane = tf.Plane(normal=[0.0, 0.0, 1.0], offset=0.0)
    >>> distances = tf.distance_field(points, plane)
    >>> distances.shape
    (1000,)
    >>>
    >>> # Distance to a segment
    >>> points_2d = np.random.rand(500, 2).astype(np.float32)
    >>> segment = tf.Segment([[0.0, 0.0], [1.0, 1.0]])
    >>> distances_2d = tf.distance_field(points_2d, segment)
    >>>
    >>> # Using with PointCloud
    >>> cloud = tf.PointCloud(points)
    >>> distances = tf.distance_field(cloud, plane)
    >>>
    >>> # Use for isocontours (example for later)
    >>> # paths, points = tf.make_isocontours(mesh, distances, [0.0, 0.5, 1.0])
    """

    # Extract points array from PointCloud if needed
    from ._spatial.point_cloud import PointCloud
    if isinstance(points, PointCloud):
        points_array = points.points
    else:
        points_array = points

    # Validate numpy array
    if not isinstance(points_array, np.ndarray):
        raise TypeError(
            f"Expected numpy array or PointCloud, got {type(points)}"
        )

    if points_array.ndim != 2:
        raise ValueError(
            f"Expected 2D array with shape (N, 2 or 3), got shape {points_array.shape}"
        )

    dims = points_array.shape[1]
    if dims not in [2, 3]:
        raise ValueError(
            f"Expected 2D or 3D points (shape (N, 2) or (N, 3)), got shape {points_array.shape}"
        )

    # Validate dtype
    if points_array.dtype not in [np.float32, np.float64]:
        raise TypeError(
            f"Points must be float32 or float64, got {points_array.dtype}. "
            f"Convert with points.astype(np.float32) or points.astype(np.float64)"
        )

    # Ensure C-contiguous
    if not points_array.flags['C_CONTIGUOUS']:
        points_array = np.ascontiguousarray(points_array)

    # Validate primitive type
    primitive_type = type(primitive)
    if primitive_type not in _DISTANCE_FIELD_DISPATCH:
        supported = ", ".join(t.__name__ for t in _DISTANCE_FIELD_DISPATCH.keys())
        raise TypeError(
            f"distance_field not implemented for primitive type: {primitive_type.__name__}. "
            f"Supported types: {supported}"
        )

    # Validate primitive has matching dims
    if not hasattr(primitive, 'dims'):
        raise TypeError("Primitive must have 'dims' attribute")

    if primitive.dims != dims:
        raise ValueError(
            f"Dimension mismatch: points have {dims}D, primitive has {primitive.dims}D. "
            f"Both must have the same dimensionality (2D or 3D)."
        )

    # Validate primitive dtype matches
    if not hasattr(primitive, 'dtype'):
        raise TypeError("Primitive must have 'dtype' attribute")

    if primitive.dtype != points_array.dtype:
        raise TypeError(
            f"Dtype mismatch: points have {points_array.dtype}, primitive has {primitive.dtype}. "
            f"Both must have the same dtype (float32 or float64)."
        )

    # Special case: Plane is 3D only
    if primitive_type is Plane and dims != 3:
        raise ValueError("distance_field with Plane is only supported in 3D")

    # Get variant suffix (e.g., "float3d" or "double2d")
    suffix = build_suffix(extract_meta(points_array))

    # Dispatch to appropriate C++ function
    func_template = _DISTANCE_FIELD_DISPATCH[primitive_type]
    func_name = func_template.format(suffix)

    return getattr(_trueform.core, func_name)(points_array, primitive.data)
