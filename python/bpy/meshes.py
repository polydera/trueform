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
from . import convert


# Session-local cache: mesh.as_pointer() -> tf.Mesh
_cache: dict[int, tf.Mesh] = {}
_CACHE_ENABLED = False
_VERBOSE = False


def _log(message: str) -> None:
    if _VERBOSE:
        print(f"[TrueformMeshCache] {message}")


def set_cache_enabled(enabled: bool) -> None:
    """
    Enable or disable the mesh cache.

    @param enabled True to enable caching, False to disable it.
    """
    global _CACHE_ENABLED
    if _CACHE_ENABLED == enabled:
        return
    _CACHE_ENABLED = enabled
    if not _CACHE_ENABLED:
        _cache.clear()
        _log("cache disabled: cleared entries")
    else:
        _log("cache enabled")


def is_cache_enabled() -> bool:
    """
    Check if the mesh cache is currently enabled.

    @return True if caching is enabled.
    """
    return _CACHE_ENABLED


def set_verbose(enabled: bool) -> None:
    """
    Enable or disable verbose cache logging.

    @param enabled True to enable verbose logging.
    """
    global _VERBOSE
    _VERBOSE = enabled


def prefetch(obj: bpy.types.Object) -> None:
    """
    Build and cache mesh structures for the given object if caching is enabled.

    @param obj Blender object of type 'MESH'.
    """
    if not _CACHE_ENABLED:
        _log("prefetch skipped (cache disabled)")
        return
    if obj.type != 'MESH':
        raise TypeError(f"Object '{obj.name}' is not a mesh")
    key = obj.data.as_pointer()
    if key in _cache:
        _log(f"prefetch cache hit: {obj.name}")
        return
    _log(f"prefetch cache miss: {obj.name}")
    _cache[key] = convert.from_blender(obj.data)


def _update(tf_mesh: tf.Mesh, mesh: bpy.types.Mesh) -> None:
    """
    Update an existing tf.Mesh with new geometry and rebuild structures.
    """
    points, faces = convert.extract_geometry(mesh)

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

    If the cache is disabled (default for script usage), a new
    transient tf.Mesh is built.

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

    if _CACHE_ENABLED:
        key = obj.data.as_pointer()
        if key not in _cache:
            _log(f"cache miss: {obj.name}")
            _cache[key] = convert.from_blender(obj.data)
        else:
            _log(f"cache hit: {obj.name}")
        mesh = _cache[key].shared_view()
    else:
        _log(f"cache disabled: building transient mesh for {obj.name}")
        mesh = convert.from_blender(obj.data)

    # Apply this object's world transform
    mesh.transformation = np.array(obj.matrix_world, dtype=np.float32)
    return mesh


@persistent
def _on_load(_):
    """Clear cache when a file is loaded."""
    _cache.clear()
    _log("cache cleared on load")


@persistent
def _on_depsgraph_update(scene, depsgraph):
    """Rebuild structures when mesh geometry changes, remove deleted meshes."""
    if not _CACHE_ENABLED:
        return
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
            _log(f"update cache: {mesh.name}")
            _update(_cache[key], mesh)
        else:
            _log(f"skip update (not cached): {mesh.name}")


def register():
    """Register handlers."""
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)

    if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)

    # Clear cache for current file
    _on_load(None)


def unregister():
    """Unregister handlers."""
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)

    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
