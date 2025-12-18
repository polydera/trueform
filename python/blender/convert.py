"""
Conversion utilities for Blender.

Create Blender meshes and objects from trueform data.

Naming follows trueform conventions:
- make_polygons / make_mesh_object: for mesh data (faces + points)
- make_curves / make_curves_object: for curve data (paths + points)

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/xlabmedical/trueform
"""

import bpy
import numpy as np

from trueform import OffsetBlockedArray


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


def make_mesh_object(faces: np.ndarray, points: np.ndarray, name: str = "Mesh") -> bpy.types.Object:
    """
    Create a Blender mesh object from numpy arrays.

    The object is added to the current scene's active collection.

    Parameters
    ----------
    faces : np.ndarray
        (M, 3) int array of triangle indices
    points : np.ndarray
        (N, 3) float array of vertex positions
    name : str
        Name for the object and mesh data block

    Returns
    -------
    bpy.types.Object
        The created Blender object
    """
    mesh = make_polygons(faces, points, name)

    obj = bpy.data.objects.new(name, mesh)
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


def make_curves_object(paths: OffsetBlockedArray, points: np.ndarray, name: str = "Curves") -> bpy.types.Object:
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
