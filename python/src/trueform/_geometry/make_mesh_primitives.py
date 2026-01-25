"""
Mesh primitive generation functions

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Tuple
import numpy as np
from .. import _trueform
from .._dispatch import dtype_str


def make_sphere_mesh(
    radius: float,
    stacks: int = 20,
    segments: int = 20,
    *,
    dtype=np.float32,
    index_dtype=np.int32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a UV sphere mesh centered at origin.

    Generates a triangulated sphere using latitude/longitude subdivision.
    All faces have outward-facing normals (CCW winding).

    Parameters
    ----------
    radius : float
        The radius of the sphere.
    stacks : int, optional
        Number of horizontal subdivisions (latitude), default 20.
    segments : int, optional
        Number of vertical subdivisions (longitude), default 20.
    dtype : numpy.dtype, optional
        Data type for vertex coordinates, default np.float32.
    index_dtype : numpy.dtype, optional
        Data type for face indices, default np.int32.

    Returns
    -------
    faces : ndarray of shape (num_faces, 3)
        Triangle face indices.
    points : ndarray of shape (num_points, 3)
        Vertex coordinates.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> faces, points = tf.make_sphere_mesh(1.0, stacks=10, segments=10)
    >>> faces.shape[1]
    3
    >>> points.shape[1]
    3
    >>> # With different dtypes
    >>> faces, points = tf.make_sphere_mesh(1.0, dtype=np.float64, index_dtype=np.int64)
    >>> points.dtype
    dtype('float64')
    """
    dtype = np.dtype(dtype)
    index_dtype = np.dtype(index_dtype)
    suffix = f"{dtype_str(index_dtype)}_{dtype_str(dtype)}"
    func = getattr(_trueform.geometry, f"make_sphere_mesh_{suffix}")
    return func(dtype.type(radius), stacks, segments)


def make_cylinder_mesh(
    radius: float,
    height: float,
    segments: int = 20,
    *,
    dtype=np.float32,
    index_dtype=np.int32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a cylinder mesh centered at origin along the z-axis.

    The cylinder extends from z = -height/2 to z = +height/2.
    All faces have outward-facing normals (CCW winding).

    Parameters
    ----------
    radius : float
        The radius of the cylinder.
    height : float
        The height of the cylinder.
    segments : int, optional
        Number of subdivisions around the circumference, default 20.
    dtype : numpy.dtype, optional
        Data type for vertex coordinates, default np.float32.
    index_dtype : numpy.dtype, optional
        Data type for face indices, default np.int32.

    Returns
    -------
    faces : ndarray of shape (num_faces, 3)
        Triangle face indices.
    points : ndarray of shape (num_points, 3)
        Vertex coordinates.

    Examples
    --------
    >>> import trueform as tf
    >>> faces, points = tf.make_cylinder_mesh(1.0, 2.0, segments=16)
    >>> faces.shape[1]
    3
    """
    dtype = np.dtype(dtype)
    index_dtype = np.dtype(index_dtype)
    suffix = f"{dtype_str(index_dtype)}_{dtype_str(dtype)}"
    func = getattr(_trueform.geometry, f"make_cylinder_mesh_{suffix}")
    return func(dtype.type(radius), dtype.type(height), segments)


def make_box_mesh(
    width: float,
    height: float,
    depth: float,
    width_ticks: int = 1,
    height_ticks: int = 1,
    depth_ticks: int = 1,
    *,
    dtype=np.float32,
    index_dtype=np.int32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an axis-aligned box mesh centered at origin.

    All faces have outward-facing normals (CCW winding).
    When tick parameters are 1, creates a simple box (8 vertices, 12 triangles).
    With higher tick values, creates a subdivided box with shared vertices.

    Parameters
    ----------
    width : float
        Size along the x-axis.
    height : float
        Size along the y-axis.
    depth : float
        Size along the z-axis.
    width_ticks : int, optional
        Number of subdivisions along x-axis, default 1.
    height_ticks : int, optional
        Number of subdivisions along y-axis, default 1.
    depth_ticks : int, optional
        Number of subdivisions along z-axis, default 1.
    dtype : numpy.dtype, optional
        Data type for vertex coordinates, default np.float32.
    index_dtype : numpy.dtype, optional
        Data type for face indices, default np.int32.

    Returns
    -------
    faces : ndarray of shape (num_faces, 3)
        Triangle face indices.
    points : ndarray of shape (num_points, 3)
        Vertex coordinates.

    Examples
    --------
    >>> import trueform as tf
    >>> # Simple box
    >>> faces, points = tf.make_box_mesh(2.0, 1.0, 3.0)
    >>> points.shape[0]
    8
    >>> faces.shape[0]
    12
    >>> # Subdivided box for simulation
    >>> faces, points = tf.make_box_mesh(2.0, 1.0, 3.0, 4, 2, 6)
    """
    dtype = np.dtype(dtype)
    index_dtype = np.dtype(index_dtype)
    suffix = f"{dtype_str(index_dtype)}_{dtype_str(dtype)}"

    if width_ticks == 1 and height_ticks == 1 and depth_ticks == 1:
        func = getattr(_trueform.geometry, f"make_box_mesh_{suffix}")
        return func(dtype.type(width), dtype.type(height), dtype.type(depth))
    else:
        func = getattr(_trueform.geometry, f"make_box_mesh_subdivided_{suffix}")
        return func(
            dtype.type(width), dtype.type(height), dtype.type(depth),
            width_ticks, height_ticks, depth_ticks
        )


def make_plane_mesh(
    width: float,
    height: float,
    width_ticks: int = 1,
    height_ticks: int = 1,
    *,
    dtype=np.float32,
    index_dtype=np.int32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a flat rectangular plane mesh in the XY plane, centered at origin.

    The plane lies at z = 0 with normal pointing +z (CCW winding).
    When tick parameters are 1, creates a simple plane (4 vertices, 2 triangles).
    With higher tick values, creates a subdivided plane.

    Parameters
    ----------
    width : float
        Size along the x-axis.
    height : float
        Size along the y-axis.
    width_ticks : int, optional
        Number of subdivisions along x-axis, default 1.
    height_ticks : int, optional
        Number of subdivisions along y-axis, default 1.
    dtype : numpy.dtype, optional
        Data type for vertex coordinates, default np.float32.
    index_dtype : numpy.dtype, optional
        Data type for face indices, default np.int32.

    Returns
    -------
    faces : ndarray of shape (num_faces, 3)
        Triangle face indices.
    points : ndarray of shape (num_points, 3)
        Vertex coordinates.

    Examples
    --------
    >>> import trueform as tf
    >>> # Simple plane
    >>> faces, points = tf.make_plane_mesh(10.0, 5.0)
    >>> points.shape[0]
    4
    >>> faces.shape[0]
    2
    >>> # Subdivided plane for deformation
    >>> faces, points = tf.make_plane_mesh(10.0, 5.0, 20, 10)
    """
    dtype = np.dtype(dtype)
    index_dtype = np.dtype(index_dtype)
    suffix = f"{dtype_str(index_dtype)}_{dtype_str(dtype)}"
    func = getattr(_trueform.geometry, f"make_plane_mesh_{suffix}")
    return func(dtype.type(width), dtype.type(height), width_ticks, height_ticks)
