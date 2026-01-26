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

    # Helper: Transform a point (affine transformation)
    def transform_point(pt):
        """Apply affine transformation: R*pt + t"""
        homogeneous = np.append(pt, 1)
        return transformation[:dims] @ homogeneous

    # Helper: Transform a vector/direction (rotation only)
    def transform_vector(vec):
        """Apply rotation only: R*vec"""
        return R @ vec

    # Dispatch based on primitive type
    if prim_type is Point:
        new_data = transform_point(primitive.data)
        return Point(new_data)

    elif prim_type is Segment:
        # Transform both endpoints
        pt0 = primitive.data[0]
        pt1 = primitive.data[1]
        new_data = np.array([
            transform_point(pt0),
            transform_point(pt1)
        ], dtype=primitive.data.dtype)
        return Segment(new_data)

    elif prim_type is Polygon:
        # Transform all vertices
        new_data = np.array([
            transform_point(pt) for pt in primitive.data
        ], dtype=primitive.data.dtype)
        return Polygon(new_data)

    elif prim_type is AABB:
        # Efficient AABB transformation without transforming all 8 corners
        # Method: Transform center and half-extents separately
        center = (primitive.data[0] + primitive.data[1]) / 2
        half_extent = (primitive.data[1] - primitive.data[0]) / 2

        # Transform center as a point
        new_center = transform_point(center)

        # Transform half-extents using absolute rotation values
        # This accounts for potential axis flipping
        new_half_extent = np.abs(R) @ half_extent

        # Reconstruct AABB
        new_min = new_center - new_half_extent
        new_max = new_center + new_half_extent
        new_data = np.array([new_min, new_max], dtype=primitive.data.dtype)
        return AABB(min=new_min, max=new_max)

    elif prim_type is Ray:
        # Transform origin (point) and direction (vector)
        origin = primitive.data[0]
        direction = primitive.data[1]

        new_origin = transform_point(origin)
        new_direction = transform_vector(direction)

        # Renormalize direction (transformation might scale it)
        new_direction = new_direction / np.linalg.norm(new_direction)

        new_data = np.array([new_origin, new_direction], dtype=primitive.data.dtype)
        return Ray(data=new_data)

    elif prim_type is Line:
        # Transform origin (point) and direction (vector)
        origin = primitive.data[0]
        direction = primitive.data[1]

        new_origin = transform_point(origin)
        new_direction = transform_vector(direction)

        # Renormalize direction (transformation might scale it)
        new_direction = new_direction / np.linalg.norm(new_direction)

        new_data = np.array([new_origin, new_direction], dtype=primitive.data.dtype)
        return Line(data=new_data)

    elif prim_type is Plane:
        # For planes, we need to transform the normal and recompute d
        # Plane equation: n·x + d = 0
        # Normal transforms as: n' = (R^-T) * n (inverse transpose for normals)
        # But for orthogonal transformations (rotation), R^-T = R

        # Get original normal and d
        normal = primitive.data[:dims]
        d = primitive.data[dims]

        # Transform normal using inverse transpose of rotation
        # For rotation matrices: R^-T = R, so we can use R directly
        # For general transformations, we'd need: (R^-1)^T
        R_inv_T = np.linalg.inv(R).T
        new_normal = R_inv_T @ normal

        # Renormalize
        new_normal = new_normal / np.linalg.norm(new_normal)

        # Recompute d from transformed normal and a point on the plane
        # Original point on plane: any point where n·p + d = 0
        # Use origin projection: p = -d * n
        point_on_plane = -d * normal
        transformed_point = transform_point(point_on_plane)
        new_d = -np.dot(new_normal, transformed_point)

        # Reconstruct plane data: [nx, ny, nz, d]
        new_data = np.append(new_normal, new_d).astype(primitive.data.dtype)
        return Plane(new_data)

    else:
        raise TypeError(
            f"transformed not implemented for type: {prim_type.__name__}. "
            f"Supported types: Point, Segment, Polygon, AABB, Ray, Line, Plane"
        )
