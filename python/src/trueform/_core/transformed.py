"""
Pure Python transformation of geometric primitives

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from typing import Union
from .._primitives import Point, Segment, Polygon, AABB, Ray, Line, Plane


def transformed(
    primitive: Union[Point, Segment, Polygon, AABB, Ray, Line, Plane],
    transformation: np.ndarray
) -> Union[Point, Segment, Polygon, AABB, Ray, Line, Plane]:
    """
    Transform a geometric primitive by a transformation matrix.

    Uses pure numpy operations for efficiency. Applies:
    - Affine transformation (rotation + translation) to points
    - Rotation only to vectors/directions

    Parameters
    ----------
    primitive : Point, Segment, Polygon, AABB, Ray, Line, or Plane
        The geometric primitive to transform
    transformation : np.ndarray
        Transformation matrix (3x3 for 2D, 4x4 for 3D)
        Format: [R | t] where R is rotation and t is translation
                [0 | 1]

    Returns
    -------
    Same type as input primitive
        A new primitive with transformed coordinates

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Create a 3D point
    >>> pt = tf.Point([1, 0, 0])
    >>>
    >>> # 90-degree rotation around Z-axis
    >>> T = np.array([
    ...     [0, -1, 0, 0],
    ...     [1,  0, 0, 0],
    ...     [0,  0, 1, 0],
    ...     [0,  0, 0, 1]
    ... ], dtype=np.float32)
    >>>
    >>> transformed_pt = tf.transformed(pt, T)
    >>> # Result: approximately [0, 1, 0]
    """
    prim_type = type(primitive)
    dims = primitive.dims

    # Validate transformation shape
    expected_shape = (dims + 1, dims + 1)
    if transformation.shape != expected_shape:
        raise ValueError(
            f"Transformation matrix must be {expected_shape} for {dims}D primitives, "
            f"got shape {transformation.shape}"
        )

    # Extract rotation part (translation handled via homogeneous coordinates)
    R = transformation[:dims, :dims]  # Rotation matrix
    dtype = primitive.data.dtype

    # Vectorized helpers that handle both single and batch shapes.
    # Any shape (..., D) is flattened to (M, D), transformed, and reshaped back.

    def transform_points(pts):
        """Apply affine transformation to point(s). Shape (..., D) -> (..., D)."""
        shape = pts.shape
        flat = pts.reshape(-1, dims)
        ones = np.ones((flat.shape[0], 1), dtype=dtype)
        homo = np.hstack([flat, ones])
        result = homo @ transformation[:dims].T
        return result.reshape(shape).astype(dtype)

    def transform_vectors(vecs):
        """Apply rotation to vector(s). Shape (..., D) -> (..., D)."""
        shape = vecs.shape
        flat = vecs.reshape(-1, dims)
        result = flat @ R.T
        return result.reshape(shape).astype(dtype)

    # Dispatch based on primitive type
    if prim_type is Point:
        # Single (D,) or batch (N, D) - all entries are points
        return Point(transform_points(primitive.data))

    elif prim_type is Segment:
        # Single (2, D) or batch (N, 2, D) - all entries are point coords
        return Segment(transform_points(primitive.data))

    elif prim_type is Polygon:
        # Single (V, D) or batch (N, V, D) - all entries are point coords
        return Polygon(transform_points(primitive.data))

    elif prim_type is AABB:
        # Single (2, D) or batch (N, 2, D)
        min_pts = primitive.data[..., 0, :]  # (..., D)
        max_pts = primitive.data[..., 1, :]  # (..., D)
        center = (min_pts + max_pts) / 2
        half_extent = (max_pts - min_pts) / 2

        new_center = transform_points(center)
        # |R|^T @ half_extent, vectorized for any leading dims
        he_shape = half_extent.shape
        new_half_extent = half_extent.reshape(-1, dims) @ np.abs(R).T
        new_half_extent = new_half_extent.reshape(he_shape).astype(dtype)

        new_min = new_center - new_half_extent
        new_max = new_center + new_half_extent
        return AABB(min=new_min, max=new_max)

    elif prim_type is Ray:
        # Single (2, D) or batch (N, 2, D)
        origin = primitive.data[..., 0, :]     # (..., D)
        direction = primitive.data[..., 1, :]  # (..., D)

        new_origin = transform_points(origin)
        new_direction = transform_vectors(direction)
        new_direction = new_direction / np.linalg.norm(
            new_direction, axis=-1, keepdims=True)

        new_data = np.stack([new_origin, new_direction], axis=-2).astype(dtype)
        return Ray(data=new_data)

    elif prim_type is Line:
        # Single (2, D) or batch (N, 2, D)
        origin = primitive.data[..., 0, :]     # (..., D)
        direction = primitive.data[..., 1, :]  # (..., D)

        new_origin = transform_points(origin)
        new_direction = transform_vectors(direction)
        new_direction = new_direction / np.linalg.norm(
            new_direction, axis=-1, keepdims=True)

        new_data = np.stack([new_origin, new_direction], axis=-2).astype(dtype)
        return Line(data=new_data)

    elif prim_type is Plane:
        # Single (D+1,) or batch (N, D+1)
        normal = primitive.data[..., :dims]  # (..., D)
        d = primitive.data[..., dims]        # scalar or (N,)

        R_inv_T = np.linalg.inv(R).T
        n_shape = normal.shape
        new_normal = normal.reshape(-1, dims) @ R_inv_T.T
        new_normal = new_normal.reshape(n_shape)
        new_normal = new_normal / np.linalg.norm(
            new_normal, axis=-1, keepdims=True)

        # Reconstruct d from a point on the original plane: p = -d * n
        point_on_plane = -np.expand_dims(d, -1) * normal
        transformed_pt = transform_points(point_on_plane)
        new_d = -np.einsum('...i,...i->...', new_normal, transformed_pt)

        new_data = np.concatenate(
            [new_normal, np.expand_dims(new_d, -1)], axis=-1).astype(dtype)
        return Plane(new_data)

    else:
        raise TypeError(
            f"transformed not implemented for type: {prim_type.__name__}. "
            f"Supported types: Point, Segment, Polygon, AABB, Ray, Line, Plane"
        )
