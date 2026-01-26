"""
Trueform Blender integration.

Provides trueform mesh operations for Blender.

Usage - Standalone scripts:
    import trueform as tf
    from trueform import bpy

    # Convert Blender mesh to tf.Mesh
    mesh = bpy.convert.from_blender(obj)

    # Run trueform operations
    result_faces, result_points = tf.boolean_difference(mesh_a, mesh_b)
    result = tf.Mesh(result_faces, result_points)

    # Convert back to Blender
    obj = bpy.convert.to_blender(result, name="Result")

Usage - Add-ons with live preview:
    from trueform import bpy

    # Register scene handlers (call once on add-on register)
    bpy.register()

    # Get tf.Mesh from scene objects (cached, with world transform)
    # Only works for objects linked to the active scene/view layer
    mesh = bpy.scene.get(obj)

    # High-level boolean operations (returns new Blender object)
    result = bpy.scene.boolean_difference(obj_a, obj_b)

    # Unregister on add-on unregister
    bpy.unregister()

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from . import convert  # noqa: F401 - re-exported as bpy.convert
from . import scene
bl_info = {
    "name": "Trueform",
    "author": "Žiga Sajovic, XLAB",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "N/A",
    "description": "Trueform mesh operations for Blender.",
    "category": "Mesh",
}


def register():
    scene.register()


def unregister():
    scene.unregister()
