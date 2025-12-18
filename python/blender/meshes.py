"""
Trueform mesh registry for Blender.

Maintains a cache of tf.Mesh instances per bpy.types.Mesh data block.
Automatically rebuilds structures when geometry changes via depsgraph.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/xlabmedical/trueform
"""

import bpy
import numpy as np
from bpy.app.handlers import persistent

import trueform as tf


# Session-local cache: mesh.as_pointer() -> tf.Mesh
_cache: dict[int, tf.Mesh] = {}


def _extract_geometry(mesh: bpy.types.Mesh) -> tuple[np.ndarray, np.ndarray]:
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


def _build(mesh: bpy.types.Mesh) -> tf.Mesh:
    """
    Build a tf.Mesh from a Blender mesh with all structures needed for boolean.
    """
    points, faces = _extract_geometry(mesh)

    m = tf.Mesh(faces, points)
    m.build_tree()
    m.build_face_membership()
    m.build_manifold_edge_link()

    return m


def _update(tf_mesh: tf.Mesh, mesh: bpy.types.Mesh) -> None:
    """
    Update an existing tf.Mesh with new geometry and rebuild structures.
    """
    points, faces = _extract_geometry(mesh)

    tf_mesh.points = points
    tf_mesh.faces = faces

    tf_mesh.build_tree()
    tf_mesh.build_face_membership()
    tf_mesh.build_manifold_edge_link()


def get(obj: bpy.types.Object) -> tf.Mesh:
    """
    Get a tf.Mesh for a Blender mesh object with its world transform.

    Returns a shared view of the cached mesh data with the object's
    world transformation applied. Multiple objects sharing the same
    mesh data will share geometry and cached structures but have
    independent transformations.

    Parameters
    ----------
    obj : bpy.types.Object
        A Blender object of type 'MESH'

    Returns
    -------
    tf.Mesh
        A shared view of the cached mesh with the object's world transform

    Raises
    ------
    TypeError
        If the object is not a mesh
    """
    if obj.type != 'MESH':
        raise TypeError(f"Object '{obj.name}' is not a mesh")

    key = obj.data.as_pointer()
    if key not in _cache:
        _cache[key] = _build(obj.data)

    # Create a shared view with this object's world transform
    mesh = _cache[key].shared_view()
    mesh.transformation = np.array(obj.matrix_world, dtype=np.float32)
    return mesh


@persistent
def _on_load(_):
    """Build cache for all meshes when a file is loaded."""
    _cache.clear()
    for mesh in bpy.data.meshes:
        _cache[mesh.as_pointer()] = _build(mesh)


@persistent
def _on_depsgraph_update(scene, depsgraph):
    """Rebuild structures when mesh geometry changes, remove deleted meshes."""
    # Remove deleted meshes from cache
    current_mesh_pointers = {mesh.as_pointer() for mesh in bpy.data.meshes}
    stale_keys = _cache.keys() - current_mesh_pointers
    for key in stale_keys:
        del _cache[key]

    # Update changed meshes
    for update in depsgraph.updates:
        if not getattr(update, "is_updated_geometry", False):
            continue

        id_ = update.id

        if isinstance(id_, bpy.types.Object) and id_.type == 'MESH':
            mesh = id_.data
        elif isinstance(id_, bpy.types.Mesh):
            mesh = id_
        else:
            continue

        key = mesh.as_pointer()
        if key in _cache:
            _update(_cache[key], mesh)
        else:
            _cache[key] = _build(mesh)


def register():
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)

    if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)

    # Build cache for current file
    _on_load(None)


def unregister():
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)

    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)

    _cache.clear()
