bl_info = {
    "name": "Trueform Boolean Tool",
    "author": "Z. Sajovic, M. Zukovec",
    "version": (1, 2),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Trueform",
    "description": "Dropdown mesh selection for Trueform Booleans",
    "category": "Mesh",
}

import bpy
import time
import sys
import importlib
import os

# --- CONFIGURATION ---
build_as_plugin = False
if build_as_plugin:
    LIB_PATH = os.path.join(os.path.dirname(__file__), "libs")
else:
    LIB_PATH = os.environ.get("TF_LIB_PATH", None)
AUTO_RELOAD = os.environ.get("TF_AUTO_RELOAD", "0") == "1"
def get_tf_libs():
    if LIB_PATH is not None and LIB_PATH not in sys.path:
        sys.path.insert(0, LIB_PATH)
    try:
        import trueform as tf
        import trueform.blender as tfb
        if AUTO_RELOAD:
            # Ensure we are using the latest version of the lib if you're developing it
            importlib.reload(tf)
            importlib.reload(tfb)
        return tf, tfb
    except ImportError:
        return None, None

def ensure_trueform_registered(tfb) -> None:
    """Register Trueform Blender operators if they are not registered yet."""
    if not hasattr(bpy.types, "TRUEFORM_OT_interactive_boolean"):
        tfb.register()

_PREVIEW_CURVES_NAME = None
_PREVIEW_MATERIAL_NAME = "Trueform_Preview_Orange"
_TF_LIBS_CACHE = None
_CACHE_HANDLERS_REGISTERED = False

def _get_tf_libs_cached():
    global _TF_LIBS_CACHE
    if AUTO_RELOAD:
        return get_tf_libs()
    if _TF_LIBS_CACHE is None:
        if LIB_PATH is not None and LIB_PATH not in sys.path:
            sys.path.insert(0, LIB_PATH)
        try:
            import trueform as tf
            import trueform.blender as tfb
        except ImportError:
            return None, None
        _TF_LIBS_CACHE = (tf, tfb)
    return _TF_LIBS_CACHE

def _get_preview_curves_obj():
    if _PREVIEW_CURVES_NAME:
        return bpy.data.objects.get(_PREVIEW_CURVES_NAME)
    return None

def _remove_preview_curves():
    global _PREVIEW_CURVES_NAME
    curves_obj = _get_preview_curves_obj()
    if curves_obj is not None:
        bpy.data.objects.remove(curves_obj, do_unlink=True)
    _PREVIEW_CURVES_NAME = None

def _tag_view3d_redraw(context):
    if context is None or context.screen is None:
        return
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()

def _style_preview_curves(curves_obj):
    curves_obj.data.bevel_depth = 0.02
    curves_obj.data.bevel_resolution = 3
    curves_obj.data.use_fill_caps = True
    curves_obj.color = (1.0, 0.55, 0.1, 1.0)
    material = bpy.data.materials.get(_PREVIEW_MATERIAL_NAME)
    if material is None:
        material = bpy.data.materials.new(name=_PREVIEW_MATERIAL_NAME)
        material.use_nodes = True
        node_tree = material.node_tree
        if node_tree is not None:
            principled = node_tree.nodes.get("Principled BSDF")
            if principled is not None:
                if "Base Color" in principled.inputs:
                    principled.inputs["Base Color"].default_value = (1.0, 0.55, 0.1, 1.0)
                if "Emission" in principled.inputs:
                    principled.inputs["Emission"].default_value = (1.0, 0.35, 0.05, 1.0)
                if "Emission Color" in principled.inputs:
                    principled.inputs["Emission Color"].default_value = (1.0, 0.35, 0.05, 1.0)
                if "Emission Strength" in principled.inputs:
                    principled.inputs["Emission Strength"].default_value = 0.6
        material.diffuse_color = (1.0, 0.55, 0.1, 1.0)
    if curves_obj.data.materials:
        curves_obj.data.materials[0] = material
    else:
        curves_obj.data.materials.append(material)
    curves_obj.active_material = material

def _update_preview_curves(tfb, paths, points):
    global _PREVIEW_CURVES_NAME
    curves_obj = _get_preview_curves_obj()
    if len(paths) > 0:
        if curves_obj is None:
            curves_obj = tfb.convert.make_curves_object(paths, points, "TFB_Preview_Curves")
            _PREVIEW_CURVES_NAME = curves_obj.name
        else:
            old_curve = curves_obj.data
            new_curve = tfb.convert.make_curves(paths, points, "TFB_Preview_Curves")
            curves_obj.data = new_curve
            bpy.data.curves.remove(old_curve)
        _style_preview_curves(curves_obj)
    else:
        if curves_obj is not None:
            bpy.data.objects.remove(curves_obj, do_unlink=True)
        _PREVIEW_CURVES_NAME = None

def _update_preview(context):
    if context is None or context.scene is None:
        return
    if not hasattr(context.scene, "trueform_tools"):
        return
    props = context.scene.trueform_tools
    if not props.interactive_preview:
        return
    tf, tfb = _get_tf_libs_cached()
    if not tf or not tfb:
        return
    ensure_trueform_registered(tfb)
    obj_a = props.target_a
    obj_b = props.target_b
    if not obj_a or not obj_b or obj_a == obj_b:
        _remove_preview_curves()
        return
    try:
        mesh_a = tfb.meshes.get(obj_a)
        mesh_b = tfb.meshes.get(obj_b)
        paths, points = tf.intersection_curves(mesh_a, mesh_b)
    except Exception as exc:
        print(f"[Trueform] Preview update failed: {exc}")
        return
    _update_preview_curves(tfb, paths, points)
    _tag_view3d_redraw(context)

def _preview_needs_update(depsgraph, obj_a, obj_b):
    for update in depsgraph.updates:
        id_ = update.id
        original = getattr(id_, "original", id_)
        if not getattr(update, "is_updated_transform", False) and not getattr(update, "is_updated_geometry", False):
            continue
        if original in {obj_a, obj_b, obj_a.data, obj_b.data}:
            return True
    return False

def _on_depsgraph_update(scene, depsgraph):
    if scene is None or not hasattr(scene, "trueform_tools"):
        return
    props = scene.trueform_tools
    if not props.interactive_preview:
        return
    obj_a = props.target_a
    obj_b = props.target_b
    if not obj_a or not obj_b or obj_a == obj_b:
        _remove_preview_curves()
        return
    if not _preview_needs_update(depsgraph, obj_a, obj_b):
        return
    _update_preview(bpy.context)

def _on_preview_toggle(self, context):
    if context is None:
        return
    if self.interactive_preview:
        tf, tfb = _get_tf_libs_cached()
        if not tf or not tfb:
            return
        ensure_trueform_registered(tfb)
        if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
        _update_preview(context)
    else:
        if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
        _remove_preview_curves()

def _on_preview_inputs_changed(self, context):
    if context is None:
        return
    _prefetch_cache(self)
    if not self.interactive_preview:
        return
    _update_preview(context)

def _create_result_collection(context, name):
    collection = bpy.data.collections.new(name)
    context.scene.collection.children.link(collection)
    return collection

def _move_object_to_collection(obj, collection):
    if obj is None or collection is None:
        return
    for existing in list(obj.users_collection):
        existing.objects.unlink(obj)
    if collection not in obj.users_collection:
        collection.objects.link(obj)

def _apply_cache_settings(props):
    tf, tfb = _get_tf_libs_cached()
    if not tf or not tfb:
        return None
    if not hasattr(tfb, "meshes"):
        return None
    if not hasattr(tfb.meshes, "set_cache_enabled"):
        return None
    tfb.meshes.set_cache_enabled(props.cache_enabled)
    tfb.meshes.set_verbose(props.cache_verbose)
    return tfb

def _prefetch_cache(props):
    global _CACHE_HANDLERS_REGISTERED
    tfb = _apply_cache_settings(props)
    if tfb is None or not props.cache_enabled:
        return
    if not _CACHE_HANDLERS_REGISTERED and hasattr(tfb.meshes, "register"):
        tfb.meshes.register()
        _CACHE_HANDLERS_REGISTERED = True
    if not hasattr(tfb.meshes, "prefetch"):
        return
    if props.target_a:
        tfb.meshes.prefetch(props.target_a)
    if props.target_b:
        tfb.meshes.prefetch(props.target_b)

def _on_cache_toggle(self, context):
    _prefetch_cache(self)

def _on_cache_verbose_toggle(self, context):
    _apply_cache_settings(self)

# --- PROPERTY GROUP ---
class TrueformProperties(bpy.types.PropertyGroup):
    target_a: bpy.props.PointerProperty(
        name="Mesh A",
        description="First mesh for boolean",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH',
        update=_on_preview_inputs_changed
    )
    target_b: bpy.props.PointerProperty(
        name="Mesh B",
        description="Second mesh for boolean",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH',
        update=_on_preview_inputs_changed
    )
    operation: bpy.props.EnumProperty(
        name="Operation",
        description="Boolean operation to apply",
        items=[
            ('DIFFERENCE', "Difference", "A - B"),
            ('UNION', "Union", "A + B"),
            ('INTERSECTION', "Intersection", "A & B"),
        ],
        default='INTERSECTION',
        update=_on_preview_inputs_changed
    )
    interactive_preview: bpy.props.BoolProperty(
        name="Interactive Preview",
        description="Show live intersection curves while adjusting inputs",
        default=False,
        update=_on_preview_toggle
    )
    cache_enabled: bpy.props.BoolProperty(
        name="Enable Mesh Cache",
        description="Precompute boolean structures when selecting inputs",
        default=False,
        update=_on_cache_toggle
    )
    cache_verbose: bpy.props.BoolProperty(
        name="Cache Verbose",
        description="Log cache hits and rebuilds to the console",
        default=False,
        update=_on_cache_verbose_toggle
    )
    hide_inputs: bpy.props.BoolProperty(
        name="Hide Inputs on Apply",
        description="Hide input meshes after creating the boolean result",
        default=False
    )

# --- OPERATOR ---
class MESH_OT_trueform_boolean(bpy.types.Operator):
    bl_idname = "mesh.trueform_boolean"
    bl_label = "Run Trueform Boolean"
    bl_description = "Calculates the boolean using the Trueform library"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        tf, tfb = get_tf_libs()
        props = context.scene.trueform_tools

        obj_a = props.target_a
        obj_b = props.target_b

        if not obj_a or not obj_b:
            self.report({'ERROR'}, "Selection incomplete: Please pick two meshes.")
            return {'CANCELLED'}

        if not tf or not tfb:
            self.report({'ERROR'}, f"Trueform library not found at {LIB_PATH}")
            return {'CANCELLED'}

        if hasattr(tfb, "meshes") and hasattr(tfb.meshes, "set_cache_enabled"):
            tfb.meshes.set_cache_enabled(props.cache_enabled)
            tfb.meshes.set_verbose(props.cache_verbose)
            if props.cache_enabled:
                tfb.meshes.prefetch(obj_a)
                tfb.meshes.prefetch(obj_b)

        start_time = time.time()

        try:
            result_name = f"TFB_{obj_a.name}_{obj_b.name}"
            curves_obj = None

            # Execute boolean
            if props.interactive_preview:
                _remove_preview_curves()
                if props.operation == 'DIFFERENCE':
                    result_obj, curves_obj = tfb.functions.boolean_difference(
                        obj_a, obj_b, name=result_name, return_curves=True
                    )
                elif props.operation == 'UNION':
                    result_obj, curves_obj = tfb.functions.boolean_union(
                        obj_a, obj_b, name=result_name, return_curves=True
                    )
                else:
                    result_obj, curves_obj = tfb.functions.boolean_intersection(
                        obj_a, obj_b, name=result_name, return_curves=True
                    )
                collection_name = f"{result_name}_Boolean"
                collection = _create_result_collection(context, collection_name)
                _move_object_to_collection(result_obj, collection)
                _move_object_to_collection(curves_obj, collection)
            else:
                if props.operation == 'DIFFERENCE':
                    result_obj = tfb.functions.boolean_difference(obj_a, obj_b, name=result_name)
                elif props.operation == 'UNION':
                    result_obj = tfb.functions.boolean_union(obj_a, obj_b, name=result_name)
                else:
                    result_obj = tfb.functions.boolean_intersection(obj_a, obj_b, name=result_name)

            # Name and organize
            result_obj.name = result_name
            if curves_obj is not None:
                curves_obj.name = f"{result_name}_Curves"

            if props.hide_inputs:
                obj_a.hide_set(True)
                obj_b.hide_set(True)

            if props.interactive_preview:
                if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
                    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
                _update_preview(context)

            self.report({'INFO'}, f"Computed in {time.time() - start_time:.4f}s")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Trueform Error: {str(e)}")
            return {'CANCELLED'}

# --- OPERATOR (INTERACTIVE PREVIEW) ---
class MESH_OT_trueform_interactive_boolean(bpy.types.Operator):
    bl_idname = "mesh.trueform_interactive_boolean"
    bl_label = "Interactive Boolean Preview"
    bl_description = "Shows live intersection curves before confirming the boolean"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        tf, tfb = get_tf_libs()
        props = context.scene.trueform_tools

        obj_a = props.target_a
        obj_b = props.target_b

        if not obj_a or not obj_b:
            self.report({'ERROR'}, "Selection incomplete: Please pick two meshes.")
            return {'CANCELLED'}

        if obj_a == obj_b:
            self.report({'ERROR'}, "Mesh A and Mesh B must be different.")
            return {'CANCELLED'}

        if not tf or not tfb:
            self.report({'ERROR'}, f"Trueform library not found at {LIB_PATH}")
            return {'CANCELLED'}

        ensure_trueform_registered(tfb)

        with context.temp_override(
            selected_objects=[obj_a, obj_b],
            selected_editable_objects=[obj_a, obj_b],
            active_object=obj_a,
        ):
            result = bpy.ops.trueform.interactive_boolean(
                'INVOKE_DEFAULT',
                operation=props.operation,
            )
        if 'CANCELLED' in result:
            self.report({'ERROR'}, "Failed to start interactive boolean.")
            return {'CANCELLED'}

        return {'FINISHED'}

# --- UI PANEL ---
class VIEW3D_PT_trueform_panel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Trueform'
    bl_label = "Boolean Setup"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.trueform_tools

        layout.label(text="Input Geometry", icon='OUTLINER_OB_MESH')

        box = layout.box()
        # The 'prop' method automatically adds the eyedropper for Object PointerProperties
        box.prop(props, "target_a")
        box.prop(props, "target_b")
        box.prop(props, "operation")
        box.prop(props, "cache_enabled")
        box.prop(props, "cache_verbose")
        box.prop(props, "interactive_preview")
        box.prop(props, "hide_inputs")

        layout.separator()

        row = layout.row(align=True)
        row.scale_y = 1.4
        row.operator("mesh.trueform_boolean", text="Apply Boolean", icon='MOD_BOOLEAN')

# --- REGISTRATION ---
classes = (
    TrueformProperties,
    MESH_OT_trueform_boolean,
    MESH_OT_trueform_interactive_boolean,
    VIEW3D_PT_trueform_panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.trueform_tools = bpy.props.PointerProperty(type=TrueformProperties)

def unregister():
    global _CACHE_HANDLERS_REGISTERED
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    _remove_preview_curves()
    _CACHE_HANDLERS_REGISTERED = False
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.trueform_tools

if __name__ == "__main__":
    register()
