"""Shared utilities for Trueform Blender addon."""

from typing import Optional
import sys
import os
import bpy

ADDON_DIR = os.path.dirname(__file__)
LIB_PATH = os.path.join(ADDON_DIR, "libs")

TEAL_MATERIAL_NAME = "Trueform_Teal"
TEAL_COLOR = (0.0, 0.835, 0.745, 1.0)


def _find_trueform_src() -> Optional[str]:
    candidates = (
        os.path.join(ADDON_DIR, "trueform"),
        os.path.join(LIB_PATH, "trueform"),
    )
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def manage_path(add=True):
    src = _find_trueform_src()
    base_paths = []
    if src:
        base_paths.append(os.path.dirname(src))

    for base in base_paths:
        if add:
            if base not in sys.path:
                sys.path.insert(0, base)
        else:
            if base in sys.path:
                sys.path.remove(base)


def get_tf_libs():
    manage_path(add=True)
    try:
        import trueform as tf
        import trueform.bpy as tfb
        return tf, tfb
    except ImportError:
        return None, None


def tag_view3d_redraw(context):
    if context and context.screen:
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def get_teal_material():
    mat = bpy.data.materials.get(TEAL_MATERIAL_NAME)
    if not mat:
        mat = bpy.data.materials.new(name=TEAL_MATERIAL_NAME)
        mat.use_nodes = True
        mat.diffuse_color = TEAL_COLOR
        nodes = mat.node_tree.nodes
        p_bsdf = nodes.get("Principled BSDF")
        if p_bsdf:
            if "Base Color" in p_bsdf.inputs:
                p_bsdf.inputs["Base Color"].default_value = TEAL_COLOR
            if "Emission Color" in p_bsdf.inputs:
                p_bsdf.inputs["Emission Color"].default_value = TEAL_COLOR
            if "Emission Strength" in p_bsdf.inputs:
                p_bsdf.inputs["Emission Strength"].default_value = 1.0
    return mat


def apply_material(obj, mat):
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat


def register():
    manage_path(add=True)
    _tf, tfb = get_tf_libs()
    if tfb:
        tfb.register()


def unregister():
    _tf, tfb = get_tf_libs()
    if tfb:
        tfb.unregister()
    manage_path(add=False)
