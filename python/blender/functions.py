"""
Trueform mesh operations for Blender.

High-level functions that take Blender objects and return Blender objects.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/xlabmedical/trueform
"""

from typing import Tuple, overload, Literal
import bpy
import trueform as tf

from . import meshes
from . import convert


# --- Boolean operations ---

@overload
def boolean_difference(
    obj_a: bpy.types.Object,
    obj_b: bpy.types.Object,
    name: str = "Boolean",
    return_curves: Literal[False] = False
) -> bpy.types.Object: ...

@overload
def boolean_difference(
    obj_a: bpy.types.Object,
    obj_b: bpy.types.Object,
    name: str = "Boolean",
    return_curves: Literal[True] = True
) -> Tuple[bpy.types.Object, bpy.types.Object]: ...

def boolean_difference(
    obj_a: bpy.types.Object,
    obj_b: bpy.types.Object,
    name: str = "Boolean",
    return_curves: bool = False
):
    """
    Compute boolean difference (A - B) between two mesh objects.

    Parameters
    ----------
    obj_a : bpy.types.Object
        First mesh object (minuend)
    obj_b : bpy.types.Object
        Second mesh object (subtrahend)
    name : str
        Name for the result object(s)
    return_curves : bool
        If True, also return intersection curves

    Returns
    -------
    bpy.types.Object or Tuple[bpy.types.Object, bpy.types.Object]
        If return_curves=False: mesh object with boolean result
        If return_curves=True: (mesh object, curves object)
    """
    mesh_a = meshes.get(obj_a)
    mesh_b = meshes.get(obj_b)

    if return_curves:
        (result_faces, result_points), labels, (paths, curve_points) = tf.boolean_difference(
            mesh_a, mesh_b, return_curves=True
        )
        mesh_obj = convert.make_mesh_object(result_faces, result_points, name)
        curves_obj = convert.make_curves_object(paths, curve_points, f"{name}_Curves")
        return mesh_obj, curves_obj
    else:
        result_faces, result_points = tf.boolean_difference(mesh_a, mesh_b)
        return convert.make_mesh_object(result_faces, result_points, name)


@overload
def boolean_union(
    obj_a: bpy.types.Object,
    obj_b: bpy.types.Object,
    name: str = "Boolean",
    return_curves: Literal[False] = False
) -> bpy.types.Object: ...

@overload
def boolean_union(
    obj_a: bpy.types.Object,
    obj_b: bpy.types.Object,
    name: str = "Boolean",
    return_curves: Literal[True] = True
) -> Tuple[bpy.types.Object, bpy.types.Object]: ...

def boolean_union(
    obj_a: bpy.types.Object,
    obj_b: bpy.types.Object,
    name: str = "Boolean",
    return_curves: bool = False
):
    """
    Compute boolean union (A ∪ B) between two mesh objects.

    Parameters
    ----------
    obj_a : bpy.types.Object
        First mesh object
    obj_b : bpy.types.Object
        Second mesh object
    name : str
        Name for the result object(s)
    return_curves : bool
        If True, also return intersection curves

    Returns
    -------
    bpy.types.Object or Tuple[bpy.types.Object, bpy.types.Object]
        If return_curves=False: mesh object with boolean result
        If return_curves=True: (mesh object, curves object)
    """
    mesh_a = meshes.get(obj_a)
    mesh_b = meshes.get(obj_b)

    if return_curves:
        (result_faces, result_points), labels, (paths, curve_points) = tf.boolean_union(
            mesh_a, mesh_b, return_curves=True
        )
        mesh_obj = convert.make_mesh_object(result_faces, result_points, name)
        curves_obj = convert.make_curves_object(paths, curve_points, f"{name}_Curves")
        return mesh_obj, curves_obj
    else:
        result_faces, result_points = tf.boolean_union(mesh_a, mesh_b)
        return convert.make_mesh_object(result_faces, result_points, name)


@overload
def boolean_intersection(
    obj_a: bpy.types.Object,
    obj_b: bpy.types.Object,
    name: str = "Boolean",
    return_curves: Literal[False] = False
) -> bpy.types.Object: ...

@overload
def boolean_intersection(
    obj_a: bpy.types.Object,
    obj_b: bpy.types.Object,
    name: str = "Boolean",
    return_curves: Literal[True] = True
) -> Tuple[bpy.types.Object, bpy.types.Object]: ...

def boolean_intersection(
    obj_a: bpy.types.Object,
    obj_b: bpy.types.Object,
    name: str = "Boolean",
    return_curves: bool = False
):
    """
    Compute boolean intersection (A ∩ B) between two mesh objects.

    Parameters
    ----------
    obj_a : bpy.types.Object
        First mesh object
    obj_b : bpy.types.Object
        Second mesh object
    name : str
        Name for the result object(s)
    return_curves : bool
        If True, also return intersection curves

    Returns
    -------
    bpy.types.Object or Tuple[bpy.types.Object, bpy.types.Object]
        If return_curves=False: mesh object with boolean result
        If return_curves=True: (mesh object, curves object)
    """
    mesh_a = meshes.get(obj_a)
    mesh_b = meshes.get(obj_b)

    if return_curves:
        (result_faces, result_points), labels, (paths, curve_points) = tf.boolean_intersection(
            mesh_a, mesh_b, return_curves=True
        )
        mesh_obj = convert.make_mesh_object(result_faces, result_points, name)
        curves_obj = convert.make_curves_object(paths, curve_points, f"{name}_Curves")
        return mesh_obj, curves_obj
    else:
        result_faces, result_points = tf.boolean_intersection(mesh_a, mesh_b)
        return convert.make_mesh_object(result_faces, result_points, name)


# --- Intersection queries ---

def intersects(obj_a: bpy.types.Object, obj_b: bpy.types.Object) -> bool:
    """
    Check if two mesh objects intersect.

    Parameters
    ----------
    obj_a : bpy.types.Object
        First mesh object
    obj_b : bpy.types.Object
        Second mesh object

    Returns
    -------
    bool
        True if the meshes intersect, False otherwise
    """
    mesh_a = meshes.get(obj_a)
    mesh_b = meshes.get(obj_b)

    return tf.intersects(mesh_a, mesh_b)


def intersection_curves(
    obj_a: bpy.types.Object,
    obj_b: bpy.types.Object,
    name: str = "Curves"
) -> bpy.types.Object:
    """
    Compute intersection curves between two mesh objects.

    Parameters
    ----------
    obj_a : bpy.types.Object
        First mesh object
    obj_b : bpy.types.Object
        Second mesh object
    name : str
        Name for the curves object

    Returns
    -------
    bpy.types.Object
        Curves object containing the intersection polylines
    """
    mesh_a = meshes.get(obj_a)
    mesh_b = meshes.get(obj_b)

    paths, points = tf.intersection_curves(mesh_a, mesh_b)

    return convert.make_curves_object(paths, points, name)
