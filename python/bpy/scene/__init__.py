"""
Trueform scene for Blender.

Maintains tf.Mesh instances per bpy.types.Mesh data block with lazy updates.
Structures rebuild on-demand when geometry changes via depsgraph.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""
from __future__ import annotations

import bpy
import numpy as np
from bpy.app.handlers import persistent

import trueform as tf
from .. import convert


# -----------------------------------------------------------------------------
# Session-local cache (mesh-key -> tf.Mesh)
# -----------------------------------------------------------------------------

_scene: dict[int, tf.Mesh] = {}
_dirty: set[int] = set()

# Mesh keys currently present in the evaluated depsgraph (our "tracked" set).
_tracked: set[int] = set()

_VERBOSE = False


def _log(message: str) -> None:
    if _VERBOSE:
        print(f"[TrueformScene] {message}")


def set_verbose(enabled: bool) -> None:
    """
    Enable or disable verbose logging.

    Parameters
    ----------
    enabled : bool
        True to enable verbose logging, False to disable.
    """
    global _VERBOSE
    _VERBOSE = bool(enabled)


def _mesh_key(mesh: bpy.types.Mesh) -> int:
    """
    Return a session-stable key for a mesh datablock.

    Parameters
    ----------
    mesh : bpy.types.Mesh
        The Blender mesh datablock.

    Returns
    -------
    int
        Session-stable identifier (session_uid if available, else as_pointer).
    """
    uid = getattr(mesh, "session_uid", None)
    if uid is not None:
        return int(uid)
    return int(mesh.as_pointer())


def _update(tf_mesh: tf.Mesh, mesh: bpy.types.Mesh) -> None:
    """
    Update an existing tf.Mesh with new geometry from a Blender mesh.

    Parameters
    ----------
    tf_mesh : tf.Mesh
        The trueform mesh to update in-place.
    mesh : bpy.types.Mesh
        The Blender mesh datablock with new geometry.
    """
    points, faces = convert.extract_geometry(mesh)

    tf_mesh.points = points
    tf_mesh.faces = faces

    # tf_mesh.build_tree()
    # tf_mesh.build_face_membership()
    # tf_mesh.build_manifold_edge_link()


def _refresh_tracked(depsgraph: bpy.types.Depsgraph) -> None:
    """
    Rebuild the set of mesh datablocks present in the depsgraph.

    Iterates evaluated objects and tracks their original mesh datablocks.

    Parameters
    ----------
    depsgraph : bpy.types.Depsgraph
        The evaluated dependency graph.
    """
    global _tracked

    tracked: set[int] = set()

    for obj_eval in depsgraph.objects:
        if obj_eval.type != 'MESH':
            continue

        obj = getattr(obj_eval, "original", None)
        if obj is None or obj.data is None:
            continue

        tracked.add(_mesh_key(obj.data))

    _tracked = tracked


def _purge_deleted_datablocks() -> None:
    """
    Purge cache entries for deleted mesh datablocks.

    Compares cached keys against current bpy.data.meshes and removes
    entries that no longer exist in the Blender data.
    """
    current = {_mesh_key(m) for m in bpy.data.meshes}
    stale = set(_scene.keys()) - current
    for key in stale:
        _scene.pop(key, None)
        _dirty.discard(key)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def get(obj: bpy.types.Object) -> tf.Mesh:
    """
    Get a tf.Mesh for a Blender mesh object with its world transform applied.

    Returns a shared view of the cached mesh. Multiple objects sharing the same
    mesh datablock share geometry and structures but have independent transforms.

    Only works for objects in the evaluated depsgraph. Objects not linked to the
    active scene/view layer (or excluded) will raise ValueError.

    Parameters
    ----------
    obj : bpy.types.Object
        A Blender object of type 'MESH'.

    Returns
    -------
    tf.Mesh
        A shared view with obj.matrix_world applied.

    Raises
    ------
    TypeError
        If the object is not a mesh.
    ValueError
        If the object has no mesh data or is not tracked by the depsgraph.
    """
    if obj.type != 'MESH':
        raise TypeError(f"Object '{obj.name}' is not a mesh")

    if obj.data is None:
        raise ValueError(f"Object '{obj.name}' has no mesh data")

    key = _mesh_key(obj.data)

    # Hard gate: only allow meshes that are present in the evaluated depsgraph
    if key not in _tracked:
        raise ValueError(
            f"Object '{obj.name}' (mesh '{obj.data.name}') is not tracked by the evaluated depsgraph. "
            f"Link it into the active scene/view layer (and ensure it's not excluded) before calling get()."
        )

    if key not in _scene:
        _log(f"building: {obj.name} (mesh='{obj.data.name}')")
        _scene[key] = convert.from_blender(obj.data)
    elif key in _dirty:
        _log(f"rebuilding (dirty): {obj.name} (mesh='{obj.data.name}')")
        _update(_scene[key], obj.data)
        _dirty.discard(key)
    else:
        _log(f"hit: {obj.name} (mesh='{obj.data.name}')")

    view = _scene[key].shared_view()
    view.transformation = np.array(obj.matrix_world, dtype=np.float32)
    return view


def clear() -> None:
    """
    Clear all cached meshes, dirty flags, and tracked state.

    Call this to reset the scene to a clean state.
    """
    _scene.clear()
    _dirty.clear()
    _tracked.clear()
    _log("cleared")


# -----------------------------------------------------------------------------
# Handlers
# -----------------------------------------------------------------------------

@persistent
def _on_load(_):
    """Clear cache when a file is loaded."""
    clear()


@persistent
def _on_depsgraph_update(scene, depsgraph):
    """
    Depsgraph callback:
      1) refresh tracked mesh set (what we can receive updates for)
      2) purge stale cache entries not tracked anymore
      3) mark tracked cached meshes dirty on geometry updates
    """
    _refresh_tracked(depsgraph)
    _purge_deleted_datablocks()
    for upd in depsgraph.updates:
        if not getattr(upd, "is_updated_geometry", False):
            continue

        if not isinstance(upd.id, bpy.types.Mesh):
            continue

        mesh = getattr(upd.id, "original", None) or upd.id
        key = _mesh_key(mesh)

        if key in _scene:
            _log(f"marking dirty (datablock changed): '{mesh.name}'")
            _dirty.add(key)


def register() -> None:
    """
    Register depsgraph handlers and initialize tracking.

    Registers load_post and depsgraph_update_post handlers, clears state,
    and primes the tracked set from the current depsgraph if available.
    """
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)

    if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)

    _on_load(None)

    try:
        dg = bpy.context.evaluated_depsgraph_get()
        _refresh_tracked(dg)
    except Exception:
        _tracked.clear()


def unregister() -> None:
    """
    Unregister depsgraph handlers and clear all state.

    Removes load_post and depsgraph_update_post handlers, and clears
    the mesh cache, dirty flags, and tracked set.
    """
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)

    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)

    clear()


# -----------------------------------------------------------------------------
# Import functions into this namespace
# -----------------------------------------------------------------------------

from .functions import *  # noqa: E402, F403
