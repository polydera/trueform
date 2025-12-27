bl_info = {
    "name": "TrueForm Mesh Cache",
    "author": "Žiga ",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "N/A",
    "description": "Session cache for TrueForm meshes, auto-build + auto-update on geometry changes.",
    "category": "Object",
}

import bpy
import numpy as np
from dataclasses import dataclass
from bpy.app.handlers import persistent

import trueform as tf


# ---------------------------------------------------------------------------
# Internal cache entry
# ---------------------------------------------------------------------------

@dataclass
class TFMeshEntry:
    """One TrueForm mesh per Blender Mesh datablock."""
    mesh_ptr: int                # mesh.as_pointer() (stable within session)
    tf_mesh: tf.Mesh             # the TrueForm mesh object
    version: int                 # our own monotonically increasing version


# Session-local cache:
#   key   = mesh.as_pointer()
#   value = TFMeshEntry
_MESH_CACHE: dict[int, TFMeshEntry] = {}

# Simple per-mesh versioning stored as a custom property
_VERSION_PROP = "_tf_geom_version"


# ---------------------------------------------------------------------------
# Low-level: read Blender mesh geometry into NumPy arrays
# ---------------------------------------------------------------------------

def read_mesh_geometry(mesh: bpy.types.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """
    Read Blender mesh vertices and faces into NumPy arrays.
    Returns (points, faces).
      - points: (N, 3) float32
      - faces:  (M, 3) int32  (triangulated)
    """
    # Ensure triangles
    mesh.calc_loop_triangles()

    # Vertices
    verts = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get("co", verts.ravel())

    # Triangles (loop_triangles.vertices is 3 indices per tri)
    tris = np.empty((len(mesh.loop_triangles), 3), dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tris.ravel())

    return verts, tris


# ---------------------------------------------------------------------------
# TrueForm construction / update
# ---------------------------------------------------------------------------

def build_tf_mesh_from_blender(mesh: bpy.types.Mesh) -> tf.Mesh:
    """
    Construct a fresh tf.Mesh from a Blender mesh.
    Builds structures needed for boolean operations.
    """
    points, faces = read_mesh_geometry(mesh)

    # Construct tf.Mesh
    m = tf.Mesh(faces, points)

    # Build structures needed for boolean
    m.build_tree()
    m.build_face_membership()
    m.build_manifold_edge_link()

    return m


def update_tf_mesh_inplace(entry: TFMeshEntry, mesh: bpy.types.Mesh) -> None:
    """
    Update an existing tf.Mesh in-place to match the Blender mesh geometry.
    Reuses the same tf.Mesh object, reassigns arrays, and rebuilds structures.
    """
    points, faces = read_mesh_geometry(mesh)

    # Update arrays (setters call mark_modified internally)
    entry.tf_mesh.points = points
    entry.tf_mesh.faces = faces

    # Rebuild structures needed for boolean
    entry.tf_mesh.build_tree()
    entry.tf_mesh.build_face_membership()
    entry.tf_mesh.build_manifold_edge_link()

    entry.version = get_mesh_version(mesh)


# ---------------------------------------------------------------------------
# Mesh versioning & cache management
# ---------------------------------------------------------------------------

def get_mesh_version(mesh: bpy.types.Mesh) -> int:
    """Read our custom 'geometry version' from the mesh datablock."""
    return int(mesh.get(_VERSION_PROP, 0))


def bump_mesh_version(mesh: bpy.types.Mesh) -> int:
    """Increment geometry version on the mesh datablock and return it."""
    v = int(mesh.get(_VERSION_PROP, 0)) + 1
    mesh[_VERSION_PROP] = v
    return v


def get_or_create_entry(mesh: bpy.types.Mesh) -> TFMeshEntry:
    """
    Ensure we have a TFMeshEntry for this mesh in the cache.
    If not present, build from scratch and store.
    """
    key = mesh.as_pointer()
    entry = _MESH_CACHE.get(key)

    if entry is None:
        # First time we see this mesh in this session
        version = get_mesh_version(mesh)  # usually 0
        tf_mesh = build_tf_mesh_from_blender(mesh)

        entry = TFMeshEntry(mesh_ptr=key, tf_mesh=tf_mesh, version=version)
        _MESH_CACHE[key] = entry

    return entry


def ensure_entry_up_to_date(mesh: bpy.types.Mesh) -> TFMeshEntry:
    """
    Ensure cache entry matches current mesh geometry.
    - If no entry: build.
    - If entry.version != mesh version: update in-place.
    """
    entry = get_or_create_entry(mesh)
    mesh_version = get_mesh_version(mesh)

    if entry.version != mesh_version:
        update_tf_mesh_inplace(entry, mesh)

    return entry


# ---------------------------------------------------------------------------
# Handlers: creation + geometry changes
# ---------------------------------------------------------------------------

@persistent
def tf_load_post(_dummy):
    """
    Called after a .blend file is loaded.
    We build cache entries for all existing meshes.
    """
    for mesh in bpy.data.meshes:
        # Ensure they have a version property (0) and a cache entry
        if _VERSION_PROP not in mesh:
            mesh[_VERSION_PROP] = 0

        get_or_create_entry(mesh)


@persistent
def tf_depsgraph_update_post(scene, depsgraph):
    """
    Called after the depsgraph updates.
    We detect:
      - newly created meshes
      - meshes whose geometry changed
    and update the cache accordingly.
    """
    for update in depsgraph.updates:
        id_ = update.id

        # Most common path: Object of type 'MESH'
        if isinstance(id_, bpy.types.Object) and id_.type == 'MESH':
            mesh = id_.data
        # Less common: direct Mesh datablock update
        elif isinstance(id_, bpy.types.Mesh):
            mesh = id_
        else:
            continue

        # New meshes or geometry changes are both handled here.
        # We treat any geometry update as a version bump.
        if getattr(update, "is_updated_geometry", False):
            new_version = bump_mesh_version(mesh)
            entry = get_or_create_entry(mesh)
            # If desired, update immediately:
            if entry.version != new_version:
                update_tf_mesh_inplace(entry, mesh)


# ---------------------------------------------------------------------------
# Public API for your other operators / tools
# ---------------------------------------------------------------------------

def get_tf_mesh_for_object(obj: bpy.types.Object) -> tf.Mesh:
    """
    Return an up-to-date tf.Mesh for the given object (MESH type).
    Geometry is synced from Blender on demand.
    """
    if obj.type != 'MESH':
        raise TypeError(f"Object {obj.name!r} is not a mesh")

    entry = ensure_entry_up_to_date(obj.data)
    return entry.tf_mesh


# ---------------------------------------------------------------------------
# Add-on register / unregister
# ---------------------------------------------------------------------------

classes = ()  # No operators here yet; just the cache system.


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # Register handlers
    if tf_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(tf_load_post)

    if tf_depsgraph_update_post not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(tf_depsgraph_update_post)

    # Build initial cache for current file (when enabling addon in an open file)
    tf_load_post(None)


def unregister():
    # Remove handlers
    if tf_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(tf_load_post)

    if tf_depsgraph_update_post in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(tf_depsgraph_update_post)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    # Clear cache
    _MESH_CACHE.clear()

