"""
Conversion utilities for Blender.

Create Blender meshes and objects from trueform data.

Naming follows trueform conventions:
- from_blender / to_blender: for mesh data (faces + points)
- make_curves / to_blender_curves: for curve data (paths + points)

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import array
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


def make_polygons(faces: np.ndarray, points: np.ndarray, name: str = "Mesh",
                  flat_shading: bool = True) -> bpy.types.Mesh:
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
    flat_shading : bool
        Whether to set faces to flat shading (default True)

    Returns
    -------
    bpy.types.Mesh
        The created Blender mesh data block
    """
    mesh = bpy.data.meshes.new(name)

    points = np.asarray(points, dtype=np.float32, order="C")
    tri_faces = np.asarray(faces, dtype=np.int32, order="C")

    n_verts = points.shape[0]
    n_tris = tri_faces.shape[0]

    # Initialize geometry
    mesh.vertices.add(n_verts)
    mesh.loops.add(n_tris * 3)
    mesh.polygons.add(n_tris)

    # Fast transfer
    mesh.vertices.foreach_set("co", points.ravel())
    mesh.loops.foreach_set("vertex_index", tri_faces.ravel())

    loop_start = np.arange(0, n_tris * 3, 3, dtype=np.int32)
    loop_total = np.full(n_tris, 3, dtype=np.int32)
    mesh.polygons.foreach_set("loop_start", loop_start)
    mesh.polygons.foreach_set("loop_total", loop_total)

    if flat_shading:
        # Explicitly set all polygons to Flat Shading (use_smooth = False)
        # This matches the default behavior of from_pydata
        mesh.polygons.foreach_set("use_smooth", np.zeros(n_tris, dtype=bool))
    else:
        mesh.polygons.foreach_set("use_smooth", np.ones(n_tris, dtype=bool))
    mesh.update(calc_edges=True)

    return mesh


def to_blender(mesh: tf.Mesh, name: str = "Mesh", flat_shading: bool = True) -> bpy.types.Object:
    """
    Create a Blender mesh object from a trueform Mesh.

    Parameters
    ----------
    mesh : tf.Mesh
        The trueform mesh to convert.
    name : str
        Name for the object and mesh data block.
    flat_shading : bool
        Whether to set faces to flat shading (default True)

    Returns
    -------
    bpy.types.Object
        The created Blender object.
    """
    points = np.array(mesh.points, copy=True)
    faces = np.array(mesh.faces, copy=False)

    mesh_data = make_polygons(faces, points, name, flat_shading)
    obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(obj)

    return obj


def _as_mv(a: np.ndarray, fmt: str) -> memoryview:
    """
    Return a memoryview over a numpy array cast to a given PEP-3118 format.
    fmt: 'f' for float32, 'i' for int32.
    Uses tobytes() to avoid platform-specific format issues ('l' vs 'i').
    """
    return memoryview(a.tobytes()).cast(fmt)


def update_blender(mesh: tf.Mesh, obj: bpy.types.Object, flat_shading: bool = True) -> None:
    """
    Update an existing Blender object's mesh with new geometry.

    Preserves object-level properties (materials, display settings, name, etc.)
    and mesh-level material slots. Clears and replaces geometry data.
    """
    points = np.asarray(mesh.points, dtype=np.float32, order="C")
    tri_faces = np.asarray(mesh.faces, dtype=np.int32, order="C")

    n_verts = points.shape[0]
    n_tris = tri_faces.shape[0]

    mesh_data = obj.data
    mesh_data.clear_geometry()

    mesh_data.vertices.add(n_verts)
    mesh_data.loops.add(n_tris * 3)
    mesh_data.polygons.add(n_tris)

    # Flatten views (no copy)
    co_flat = points.reshape(-1)
    vi_flat = tri_faces.reshape(-1)

    # Precompute polygon arrays (triangles)
    loop_start = np.arange(n_tris, dtype=np.int32) * 3
    loop_total = np.full(n_tris, 3, dtype=np.int32)

    # Use memoryview via tobytes for reliable cross-platform data transfer
    try:
        mesh_data.vertices.foreach_set("co", _as_mv(co_flat, 'f'))
        mesh_data.loops.foreach_set("vertex_index", _as_mv(vi_flat, 'i'))
        mesh_data.polygons.foreach_set("loop_start", _as_mv(loop_start, 'i'))
        mesh_data.polygons.foreach_set("loop_total", _as_mv(loop_total, 'i'))
    except Exception:
        # Fallback: array.array buffers
        co_buf = array.array('f')
        co_buf.frombytes(co_flat.tobytes())
        mesh_data.vertices.foreach_set("co", co_buf)

        vi_buf = array.array('i')
        vi_buf.frombytes(vi_flat.tobytes())
        mesh_data.loops.foreach_set("vertex_index", vi_buf)

        ls_buf = array.array('i')
        ls_buf.frombytes(loop_start.tobytes())
        mesh_data.polygons.foreach_set("loop_start", ls_buf)

        lt_buf = array.array('i')
        lt_buf.frombytes(loop_total.tobytes())
        mesh_data.polygons.foreach_set("loop_total", lt_buf)

    # Flat shading is default; only set smooth if requested
    if not flat_shading:
        sm = np.ones(n_tris, dtype=np.bool_)
        try:
            mesh_data.polygons.foreach_set("use_smooth", memoryview(sm).cast('b'))
        except Exception:
            sm_buf = array.array('b')
            sm_buf.frombytes(sm.tobytes())
            mesh_data.polygons.foreach_set("use_smooth", sm_buf)

    mesh_data.update(calc_edges=True)


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


def update_curves(paths: OffsetBlockedArray, points: np.ndarray, obj: bpy.types.Object) -> None:
    """
    Update an existing Blender curve object with new paths and points.

    Preserves curve-level settings (bevel_depth, materials, etc.)
    by clearing and refilling splines in place.

    Parameters
    ----------
    paths : OffsetBlockedArray
        Paths as indices into the points array. Each path is one polyline.
    points : np.ndarray
        (N, 3) float array of point coordinates
    obj : bpy.types.Object
        The existing Blender curve object to update.
    """
    curve = obj.data

    # Clear existing splines
    curve.splines.clear()

    # Add new splines
    for path_indices in paths:
        path_points = points[path_indices]
        n_points = len(path_points)

        spline = curve.splines.new('POLY')
        spline.points.add(n_points - 1)

        coords = np.empty((n_points, 4), dtype=np.float32)
        coords[:, :3] = path_points
        coords[:, 3] = 1.0  # w component
        spline.points.foreach_set("co", coords.ravel())


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
