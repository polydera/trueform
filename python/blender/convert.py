"""
Conversion utilities for Blender.

Create Blender meshes and objects from trueform data.

Naming follows trueform conventions:
- from_blender / to_blender: for mesh data (faces + points)
- make_curves / to_blender_curves: for curve data (paths + points)

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/xlabmedical/trueform
"""

import bpy
import numpy as np
from typing import Union

import trueform as tf
from trueform import OffsetBlockedArray


def extract_geometry(mesh: bpy.types.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract vertices and triangulated faces from a Blender mesh.

    Returns
    -------
    points : np.ndarray
        (N, 3) float32 array of vertex positions
    faces : np.ndarray
        (M, 3) int32 array of triangle indices
    """
    mesh.calc_loop_triangles()

    points = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get("co", points.ravel())

    faces = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", faces.ravel())

    return points, faces


def _apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix is None:
        return points
    ones = np.ones((len(points), 1), dtype=points.dtype)
    homogeneous = np.concatenate((points, ones), axis=1)
    transformed = (matrix @ homogeneous.T).T
    return transformed[:, :3].astype(points.dtype, copy=False)


def from_blender(data: Union[bpy.types.Mesh, bpy.types.Object]) -> tf.Mesh:
    """
    Convert a Blender mesh or object to a trueform Mesh.

    Automatically builds spatial tree, face membership and manifold edge links
    required for most operations (e.g. boolean). If the input is an object,
    its world transform is baked into the resulting points.

    Parameters
    ----------
    data : bpy.types.Mesh or bpy.types.Object
        The Blender mesh data block or object to convert.

    Returns
    -------
    tf.Mesh
        The trueform Mesh representation.
    """
    matrix = None
    if isinstance(data, bpy.types.Object):
        if data.type != 'MESH':
            raise TypeError(f"Object '{data.name}' is not a mesh")
        mesh_data = data.data
        matrix = np.array(data.matrix_world, dtype=np.float32)
    else:
        mesh_data = data

    points, faces = extract_geometry(mesh_data)

    if matrix is not None:
        points = _apply_transform(points, matrix)

    mesh = tf.Mesh(faces, points)

    # Build production-ready structures
    mesh.build_tree()
    mesh.build_face_membership()
    mesh.build_manifold_edge_link()

    return mesh


def make_polygons(faces: np.ndarray, points: np.ndarray, name: str = "Mesh") -> bpy.types.Mesh:
    """
    Create a Blender mesh data block from numpy arrays.

    Parameters
    ----------
    faces : np.ndarray
        (M, 3) int array of triangle indices
    points : np.ndarray
        (N, 3) float array of vertex positions
    name : str
        Name for the mesh data block

    Returns
    -------
    bpy.types.Mesh
        The created Blender mesh data block
    """
    mesh = bpy.data.meshes.new(name)

    vertices = points.tolist()
    triangles = faces.tolist()

    mesh.from_pydata(vertices, [], triangles)
    mesh.update()

    return mesh


def to_blender(mesh: tf.Mesh, name: str = "Mesh") -> bpy.types.Object:
    """
    Create a Blender mesh object from a trueform Mesh.

    Parameters
    ----------
    mesh : tf.Mesh
        The trueform mesh to convert.
    name : str
        Name for the object and mesh data block.

    Returns
    -------
    bpy.types.Object
        The created Blender object.
    """
    points = np.array(mesh.points, copy=True)
    faces = np.array(mesh.faces, copy=False)

    mesh_data = make_polygons(faces, points, name)
    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)

    return obj


def make_curves(paths: OffsetBlockedArray, points: np.ndarray, name: str = "Curves") -> bpy.types.Curve:
    """
    Create a Blender curve data block from paths and points.

    Each path becomes a separate polyline spline in the curve.

    Parameters
    ----------
    paths : OffsetBlockedArray
        Paths as indices into the points array. Each path is one polyline.
    points : np.ndarray
        (N, 3) float array of point coordinates

    name : str
        Name for the curve data block

    Returns
    -------
    bpy.types.Curve
        The created Blender curve data block
    """
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'

    for path_indices in paths:
        path_points = points[path_indices]
        n_points = len(path_points)

        spline = curve.splines.new('POLY')
        spline.points.add(n_points - 1)

        coords = np.empty((n_points, 4), dtype=np.float32)
        coords[:, :3] = path_points
        coords[:, 3] = 1.0  # w component
        spline.points.foreach_set("co", coords.ravel())

    return curve


def to_blender_curves(paths: OffsetBlockedArray, points: np.ndarray, name: str = "Curves") -> bpy.types.Object:
    """
    Create a Blender curve object from paths and points.

    The object is added to the current scene's active collection.

    Parameters
    ----------
    paths : OffsetBlockedArray
        Paths as indices into the points array. Each path is one polyline.
    points : np.ndarray
        (N, 3) float array of point coordinates
    name : str
        Name for the object and curve data block

    Returns
    -------
    bpy.types.Object
        The created Blender object
    """
    curve = make_curves(paths, points, name)

    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)

    return obj
