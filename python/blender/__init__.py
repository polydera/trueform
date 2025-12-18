"""
Trueform Blender integration.

Provides trueform mesh operations for Blender.

Usage:
    from trueform import blender

    # Get tf.Mesh from Blender object (cached, with world transform)
    mesh = blender.meshes.get(obj)

    # Boolean operations (returns new Blender object)
    result = blender.functions.boolean_difference(obj_a, obj_b)
    result = blender.functions.boolean_union(obj_a, obj_b)
    result = blender.functions.boolean_intersection(obj_a, obj_b)

    # Check intersection
    if blender.functions.intersects(obj_a, obj_b):
        ...

    # Create Blender object from arrays
    obj = blender.convert.create_object(faces, points, name="Result")

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/xlabmedical/trueform
"""

bl_info = {
    "name": "Trueform",
    "author": "Žiga Sajovic, XLAB",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "N/A",
    "description": "Trueform mesh operations for Blender.",
    "category": "Mesh",
}

from . import meshes
from . import convert
from . import functions
from . import operators


def register():
    meshes.register()
    operators.register()


def unregister():
    operators.unregister()
    meshes.unregister()
