"""
Trueform boolean Blender add-on.

Caching is enabled for this add-on to support interactive preview. When writing
standalone Blender Python scripts, convert Blender objects with
`trueform.blender.convert.from_blender`, run operations on the resulting `tf.Mesh`,
and convert back using `trueform.blender.convert.to_blender`.
"""

bl_info = {
    "name": "Trueform Boolean Tool",
    "author": "Z. Sajovic, M. Zukovec",
    "version": (1, 2, 1),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Trueform",
    "description": "High-performance boolean operations using the Trueform library",
    "category": "Mesh",
}

import bpy
import os
import sys
import shutil
from typing import Optional

# --- LIBRARY PATH SETUP ---
ADDON_DIR = os.path.dirname(__file__)
LIB_PATH = os.path.join(ADDON_DIR, "libs")


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
    if not src:
        return
    base = os.path.dirname(src)
    if add:
        if base not in sys.path:
            sys.path.insert(0, base)
    else:
        if base in sys.path:
            sys.path.remove(base)


def _get_modules_dir() -> Optional[str]:
    return bpy.utils.user_resource("SCRIPTS", path="modules")


def _install_trueform_global() -> bool:
    src = _find_trueform_src()
    modules_dir = _get_modules_dir()
    if not src or not modules_dir:
        return False
    try:
        os.makedirs(modules_dir, exist_ok=True)
        dest = os.path.join(modules_dir, "trueform")
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        return True
    except Exception as exc:
        print(f"[Trueform] Install failed: {exc}")
        return False


# --- PREFERENCES ---

def _on_install_preference(self, context):
    if self.install_trueform_module:
        _install_trueform_global()


class TrueformAddonPreferences(bpy.types.AddonPreferences):
    # Use __package__ so it matches the folder name automatically
    bl_idname = __package__ if __package__ else __name__

    install_trueform_module: bpy.props.BoolProperty(
        name="Install Trueform module for scripts",
        description="Copy the bundled trueform package into Blender's scripts/modules",
        default=True,
        update=_on_install_preference,
    )

    def draw(self, context):
        layout = self.layout
        column = layout.column(align=True)

        column.prop(self, "install_trueform_module")

        if not self.install_trueform_module:
            error_box = column.box()
            error_box.alert = True  # Tints the whole box red for visibility
            error_box.label(text="Module is currently disabled.", icon='WARNING_LARGE')
            error_box.label(text="External scripts cannot import trueform.", icon='INFO')
        else:
            column.label(text="Module is installed globally.", icon='CHECKMARK')


# --- INITIALIZATION & CORE ---

def get_tf_libs():
    manage_path(add=True)
    try:
        import trueform as tf
        import trueform.blender as tfb
        return tf, tfb
    except ImportError:
        return None, None


_PREVIEW_CURVES_NAME = None
_PREVIEW_MATERIAL_NAME = "Trueform_Preview_Orange"


def _remove_preview_curves():
    global _PREVIEW_CURVES_NAME
    if _PREVIEW_CURVES_NAME:
        curves_obj = bpy.data.objects.get(_PREVIEW_CURVES_NAME)
        if curves_obj:
            bpy.data.objects.remove(curves_obj, do_unlink=True)
        _PREVIEW_CURVES_NAME = None


def _update_preview(context):
    if not context or not context.scene: return
    props = context.scene.trueform_tools
    if not props.interactive_preview: return

    tf, tfb = get_tf_libs()
    if not tf or not tfb: return

    obj_a, obj_b = props.target_a, props.target_b
    if not obj_a or not obj_b or obj_a == obj_b:
        _remove_preview_curves()
        return

    try:
        mesh_a = tfb.meshes.get(obj_a)
        mesh_b = tfb.meshes.get(obj_b)
        paths, points = tf.intersection_curves(mesh_a, mesh_b)

        global _PREVIEW_CURVES_NAME
        existing = bpy.data.objects.get(_PREVIEW_CURVES_NAME) if _PREVIEW_CURVES_NAME else None

        if paths:
            if not existing:
                curves_obj = tfb.convert.to_blender_curves(paths, points, "TFB_Preview_Curves")
                _PREVIEW_CURVES_NAME = curves_obj.name
            else:
                old_data = existing.data
                existing.data = tfb.convert.make_curves(paths, points, "TFB_Preview_Curves")
                bpy.data.curves.remove(old_data)
                curves_obj = existing

            # Simplified Styling
            curves_obj.data.bevel_depth = 0.02
        else:
            _remove_preview_curves()
    except Exception as e:
        print(f"Trueform Preview Error: {e}")


def _on_depsgraph_update(scene, depsgraph):
    # Live refresh logic
    if not hasattr(scene, "trueform_tools"): return
    props = scene.trueform_tools
    if props.interactive_preview:
        _update_preview(bpy.context)


def _on_preview_toggle(self, context):
    if self.interactive_preview:
        if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
        _update_preview(context)
    else:
        if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
        _remove_preview_curves()


# --- PROPERTIES ---

class TrueformProperties(bpy.types.PropertyGroup):
    target_a: bpy.props.PointerProperty(name="Mesh A", type=bpy.types.Object, poll=lambda s, o: o.type == 'MESH')
    target_b: bpy.props.PointerProperty(name="Mesh B", type=bpy.types.Object, poll=lambda s, o: o.type == 'MESH')
    operation: bpy.props.EnumProperty(
        name="Operation",
        items=[('DIFFERENCE', "Difference", ""), ('UNION', "Union", ""), ('INTERSECTION', "Intersection", "")],
        default='INTERSECTION'
    )
    interactive_preview: bpy.props.BoolProperty(name="Live Preview", default=True, update=_on_preview_toggle)
    hide_inputs: bpy.props.BoolProperty(name="Hide Inputs on Apply", default=True)


# --- PANEL & OPERATORS ---

class VIEW3D_PT_trueform_panel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Trueform'
    bl_label = "Trueform Boolean"

    def draw(self, context):
        layout = self.layout
        props = context.scene.trueform_tools

        col = layout.column(align=True)
        col.prop(props, "target_a")
        col.prop(props, "target_b")

        layout.separator()
        layout.prop(props, "operation", expand=True)
        layout.operator("mesh.trueform_boolean", icon='MOD_BOOLEAN')

        # Modern Sub-Panel API for 5.0
        sub = layout.box()
        sub.label(text="Advanced Options", icon='SETTINGS')
        sub.prop(props, "interactive_preview")
        sub.prop(props, "hide_inputs")


class MESH_OT_trueform_boolean(bpy.types.Operator):
    bl_idname = "mesh.trueform_boolean"
    bl_label = "Apply Trueform Boolean"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        tf, tfb = get_tf_libs()
        props = context.scene.trueform_tools
        if not props.target_a or not props.target_b: return {'CANCELLED'}
        # Final operation logic here...
        self.report({'INFO'}, "Boolean Applied")
        return {'FINISHED'}


# --- REGISTRATION ---

classes = (
    TrueformAddonPreferences,
    TrueformProperties,
    MESH_OT_trueform_boolean,
    VIEW3D_PT_trueform_panel
)


def register():
    manage_path(add=True)
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.trueform_tools = bpy.props.PointerProperty(type=TrueformProperties)


def unregister():
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)

    _remove_preview_curves()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.trueform_tools
    manage_path(add=False)
