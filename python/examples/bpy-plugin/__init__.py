"""
Trueform boolean Blender add-on.

Caching is enabled for this add-on to support interactive preview. When writing
standalone Blender Python scripts, convert Blender objects with
`trueform.blender.convert.from_blender`, run operations on the resulting `tf.Mesh`,
and convert back using `trueform.blender.convert.to_blender`.
"""

from typing import Optional
import sys
import os
import bpy
bl_info = {
    "name": "Trueform Boolean Tool",
    "author": "Z. Sajovic, M. Zukovec",
    "version": (1, 2, 1),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Trueform",
    "description": "High-performance boolean operations using the Trueform library",
    "category": "Mesh",
}


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

# --- INITIALIZATION ---


def get_tf_libs():
    manage_path(add=True)
    try:
        import trueform as tf
        import trueform.bpy as tfb
        return tf, tfb
    except ImportError:
        return None, None


_PREVIEW_CURVES_NAME = None
_PREVIEW_MATERIAL_NAME = "Trueform_Preview_Orange"

# --- UTILITIES ---


def _remove_preview_curves():
    global _PREVIEW_CURVES_NAME
    if _PREVIEW_CURVES_NAME:
        curves_obj = bpy.data.objects.get(_PREVIEW_CURVES_NAME)
        if curves_obj:
            bpy.data.objects.remove(curves_obj, do_unlink=True)
        _PREVIEW_CURVES_NAME = None


def _tag_view3d_redraw(context):
    if context and context.screen:
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _style_preview_curves(curves_obj):
    curves_obj.data.bevel_depth = 0.02
    curves_obj.data.bevel_resolution = 3
    curves_obj.data.use_fill_caps = True

    mat = bpy.data.materials.get(_PREVIEW_MATERIAL_NAME)
    if not mat:
        mat = bpy.data.materials.new(name=_PREVIEW_MATERIAL_NAME)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        p_bsdf = nodes.get("Principled BSDF")
        if p_bsdf:
            color = (1.0, 0.55, 0.1, 1.0)
            if "Base Color" in p_bsdf.inputs:
                p_bsdf.inputs["Base Color"].default_value = color
            if "Emission Color" in p_bsdf.inputs:
                p_bsdf.inputs["Emission Color"].default_value = color
            if "Emission Strength" in p_bsdf.inputs:
                p_bsdf.inputs["Emission Strength"].default_value = 1.0

    if not curves_obj.data.materials:
        curves_obj.data.materials.append(mat)
    else:
        curves_obj.data.materials[0] = mat

# --- REAL-TIME ENGINE ---


def _update_preview(context):
    if not context or not context.scene:
        return
    props = context.scene.trueform_tools
    if not props.interactive_preview:
        return

    tf, tfb = get_tf_libs()
    if not tf or not tfb:
        return

    obj_a, obj_b = props.target_a, props.target_b

    if not obj_a or not obj_b or obj_a == obj_b:
        _remove_preview_curves()
        return

    try:
        mesh_a = tfb.scene.get(obj_a)
        mesh_b = tfb.scene.get(obj_b)
        paths, points = tf.intersection_curves(mesh_a, mesh_b)

        global _PREVIEW_CURVES_NAME
        existing = bpy.data.objects.get(
            _PREVIEW_CURVES_NAME) if _PREVIEW_CURVES_NAME else None

        if paths:
            if not existing:
                curves_obj = tfb.convert.to_blender_curves(
                    paths, points, "TFB_Preview_Curves")
                _PREVIEW_CURVES_NAME = curves_obj.name
            else:
                old_data = existing.data
                existing.data = tfb.convert.make_curves(
                    paths, points, "TFB_Preview_Curves")
                bpy.data.curves.remove(old_data)
                curves_obj = existing
            _style_preview_curves(curves_obj)
        else:
            _remove_preview_curves()

        _tag_view3d_redraw(context)
    except Exception as e:
        print(f"Trueform Preview Error: {e}")


def _on_depsgraph_update(scene, depsgraph):
    if not hasattr(scene, "trueform_tools"):
        return
    props = scene.trueform_tools
    if not props.interactive_preview:
        return

    targets = {props.target_a, props.target_b}
    # Check if geometry or transform of targets changed
    for upd in depsgraph.updates:
        if upd.id.original in targets or (hasattr(upd.id, "data") and upd.id.data in [t.data for t in targets if t]):
            _update_preview(bpy.context)
            break


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
    target_a: bpy.props.PointerProperty(name="Mesh A", type=bpy.types.Object,
                                        poll=lambda s, o: o.type == 'MESH', update=lambda s, c: _update_preview(c))
    target_b: bpy.props.PointerProperty(name="Mesh B", type=bpy.types.Object,
                                        poll=lambda s, o: o.type == 'MESH', update=lambda s, c: _update_preview(c))
    operation: bpy.props.EnumProperty(
        name="Operation",
        items=[('DIFFERENCE', "Difference", ""), ('UNION', "Union", ""),
               ('INTERSECTION', "Intersection", "")],
        default='INTERSECTION', update=lambda s, c: _update_preview(c)
    )
    interactive_preview: bpy.props.BoolProperty(
        name="Live Preview", default=True, update=_on_preview_toggle)
    hide_inputs: bpy.props.BoolProperty(
        name="Hide Inputs on Apply", default=True)

# --- OPERATORS ---


class MESH_OT_trueform_boolean(bpy.types.Operator):
    bl_idname = "mesh.trueform_boolean"
    bl_label = "Apply Trueform Boolean"
    bl_description = "Calculate final boolean"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        tf, tfb = get_tf_libs()
        props = context.scene.trueform_tools
        if not props.target_a or not props.target_b:
            return {'CANCELLED'}

        _remove_preview_curves()
        try:
            op_map = {'DIFFERENCE': tfb.scene.boolean_difference,
                      'UNION': tfb.scene.boolean_union, 'INTERSECTION': tfb.scene.boolean_intersection}
            _result = op_map[props.operation](
                props.target_a, props.target_b, name=f"TFB_{props.target_a.name}")
            if props.hide_inputs:
                props.target_a.hide_set(True)
                props.target_b.hide_set(True)
            _remove_preview_curves()
            props.target_a = None
            props.target_b = None
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}


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
        layout.prop(props, "operation", expand=True)
        layout.operator("mesh.trueform_boolean", icon='MOD_BOOLEAN')

        # Safe Panel Drawing
        p_tuple = layout.panel("tf_adv", default_closed=True)
        if p_tuple:
            header, body = p_tuple
            header.label(text="Advanced")
            if body:
                body.prop(props, "interactive_preview")
                body.prop(props, "hide_inputs")


# --- REGISTRATION ---
classes = (TrueformProperties, MESH_OT_trueform_boolean,
           VIEW3D_PT_trueform_panel)


def register():
    manage_path(add=True)
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.trueform_tools = bpy.props.PointerProperty(
        type=TrueformProperties)

    _tf, tfb = get_tf_libs()
    if tfb:
        tfb.register()

    if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)


def unregister():
    _tf, tfb = get_tf_libs()
    if tfb:
        tfb.unregister()

    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    _remove_preview_curves()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.trueform_tools
    manage_path(add=False)


if __name__ == "__main__":
    register()
