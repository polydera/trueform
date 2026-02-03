"""
Boolean modifier tool for Trueform Blender addon.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import bpy
from .. import core


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _is_trueform_result(obj) -> bool:
    """Check if object is a Trueform boolean result."""
    if not obj or obj.type != 'MESH':
        return False
    mod = getattr(obj, 'trueform_boolean', None)
    return mod is not None and mod.source_a is not None


def _mesh_poll(self, obj):
    """Poll function for source selection - must be mesh."""
    return obj.type == 'MESH'


def _get_live_results():
    """Get all result objects with live=True."""
    results = []
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        mod = getattr(obj, 'trueform_boolean', None)
        if mod and mod.live and mod.source_a and mod.source_b:
            results.append(obj)
    return results


# -----------------------------------------------------------------------------
# Update Logic
# -----------------------------------------------------------------------------

def _update_result(result_obj):
    """Update a single result object from its sources."""
    mod = result_obj.trueform_boolean
    if not mod or not mod.source_a or not mod.source_b:
        return
    if mod.source_a == mod.source_b:
        return

    tf, tfb = core.get_tf_libs()
    if not tf or not tfb:
        return

    try:
        op_map = {
            'DIFFERENCE': tfb.scene.boolean_difference_mesh,
            'UNION': tfb.scene.boolean_union_mesh,
            'INTERSECTION': tfb.scene.boolean_intersection_mesh
        }
        result_mesh = op_map[mod.operation](mod.source_a, mod.source_b)
        tfb.convert.update_blender(result_mesh, result_obj)
        core.tag_view3d_redraw(bpy.context)
    except Exception as e:
        print(f"Trueform Boolean Error on '{result_obj.name}': {e}")


# -----------------------------------------------------------------------------
# Handlers
# -----------------------------------------------------------------------------

def _on_depsgraph_update(scene, depsgraph):
    """Handle depsgraph updates - update live results when sources change."""
    live_results = _get_live_results()
    if not live_results:
        return

    # Check which objects/meshes were updated
    for result_obj in live_results:
        mod = result_obj.trueform_boolean
        targets = {mod.source_a, mod.source_b}
        target_data = {t.data for t in targets if t and t.data}

        for upd in depsgraph.updates:
            upd_id = getattr(upd.id, 'original', upd.id)
            if upd_id in targets:
                _update_result(result_obj)
                break
            if hasattr(upd.id, 'data') and upd.id.data in target_data:
                _update_result(result_obj)
                break


def _on_frame_change(scene, depsgraph):
    """Handle frame changes - update all live results."""
    for result_obj in _get_live_results():
        _update_result(result_obj)


def _ensure_handlers():
    """Add handlers if any live results exist."""
    if _get_live_results():
        if _on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
        if _on_frame_change not in bpy.app.handlers.frame_change_post:
            bpy.app.handlers.frame_change_post.append(_on_frame_change)
    else:
        _remove_handlers()


def _remove_handlers():
    """Remove handlers."""
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    if _on_frame_change in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_on_frame_change)


# -----------------------------------------------------------------------------
# Property Callbacks
# -----------------------------------------------------------------------------

def _on_modifier_update(self, context):
    """Called when source_a, source_b, or operation changes on a result."""
    obj = context.active_object
    if obj and _is_trueform_result(obj) and self.live:
        _update_result(obj)


def _on_live_toggle(self, context):
    """Called when live toggle changes on a result."""
    obj = context.active_object
    if not obj:
        return

    if self.live:
        _ensure_handlers()
        _update_result(obj)
    else:
        _ensure_handlers()  # Will remove if no live results left


# -----------------------------------------------------------------------------
# Property Groups
# -----------------------------------------------------------------------------

class TrueformBooleanModifier(bpy.types.PropertyGroup):
    """Modifier data stored on result objects."""
    source_a: bpy.props.PointerProperty(
        name="Source A",
        type=bpy.types.Object,
        poll=_mesh_poll,
        update=_on_modifier_update
    )
    source_b: bpy.props.PointerProperty(
        name="Source B",
        type=bpy.types.Object,
        poll=_mesh_poll,
        update=_on_modifier_update
    )
    operation: bpy.props.EnumProperty(
        name="Operation",
        items=[
            ('DIFFERENCE', "Difference", "A - B"),
            ('UNION', "Union", "A ∪ B"),
            ('INTERSECTION', "Intersection", "A ∩ B"),
        ],
        default='DIFFERENCE',
        update=_on_modifier_update
    )
    live: bpy.props.BoolProperty(
        name="Live",
        default=True,
        update=_on_live_toggle
    )


class TrueformBooleanCreateProps(bpy.types.PropertyGroup):
    """Scene-level properties for creation UI."""
    target_a: bpy.props.PointerProperty(
        name="Mesh A",
        type=bpy.types.Object,
        poll=_mesh_poll
    )
    target_b: bpy.props.PointerProperty(
        name="Mesh B",
        type=bpy.types.Object,
        poll=_mesh_poll
    )
    operation: bpy.props.EnumProperty(
        name="Operation",
        items=[
            ('DIFFERENCE', "Difference", "A - B"),
            ('UNION', "Union", "A ∪ B"),
            ('INTERSECTION', "Intersection", "A ∩ B"),
        ],
        default='DIFFERENCE'
    )


# -----------------------------------------------------------------------------
# Operators
# -----------------------------------------------------------------------------

class MESH_OT_trueform_boolean_create(bpy.types.Operator):
    """Create a new Trueform boolean from two meshes"""
    bl_idname = "mesh.trueform_boolean_create"
    bl_label = "Create Boolean"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.trueform_boolean_create

        if not props.target_a or not props.target_b:
            self.report({'WARNING'}, "Select two meshes")
            return {'CANCELLED'}

        if props.target_a == props.target_b:
            self.report({'WARNING'}, "Select two different meshes")
            return {'CANCELLED'}

        tf, tfb = core.get_tf_libs()
        if not tf or not tfb:
            self.report({'ERROR'}, "Trueform library not loaded")
            return {'CANCELLED'}

        try:
            # Compute initial result
            op_map = {
                'DIFFERENCE': tfb.scene.boolean_difference_mesh,
                'UNION': tfb.scene.boolean_union_mesh,
                'INTERSECTION': tfb.scene.boolean_intersection_mesh
            }
            result_mesh = op_map[props.operation](props.target_a, props.target_b)
            result_obj = tfb.convert.to_blender(result_mesh, name=f"TF_{props.target_a.name}")

            # Store modifier data on result
            mod = result_obj.trueform_boolean
            mod.source_a = props.target_a
            mod.source_b = props.target_b
            mod.operation = props.operation
            mod.live = True

            # Register handlers
            _ensure_handlers()

            # Select result
            bpy.ops.object.select_all(action='DESELECT')
            result_obj.select_set(True)
            context.view_layer.objects.active = result_obj

            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}


class MESH_OT_trueform_boolean_refresh(bpy.types.Operator):
    """Force refresh the boolean result"""
    bl_idname = "mesh.trueform_boolean_refresh"
    bl_label = "Refresh"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and _is_trueform_result(context.active_object)

    def execute(self, context):
        _update_result(context.active_object)
        return {'FINISHED'}


class MESH_OT_trueform_boolean_remove(bpy.types.Operator):
    """Remove Trueform boolean modifier, keeping result as static mesh"""
    bl_idname = "mesh.trueform_boolean_remove"
    bl_label = "Remove Modifier"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and _is_trueform_result(context.active_object)

    def execute(self, context):
        obj = context.active_object
        mod = obj.trueform_boolean

        # Clear modifier data
        mod.source_a = None
        mod.source_b = None
        mod.live = False

        # Update handlers
        _ensure_handlers()

        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Panel
# -----------------------------------------------------------------------------

class VIEW3D_PT_trueform_boolean(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Trueform'
    bl_label = "Boolean"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        if obj and _is_trueform_result(obj):
            # MODIFIER MODE: Show result's settings
            mod = obj.trueform_boolean

            # Warning if sources missing
            if not mod.source_a or not mod.source_b:
                layout.label(text="Source missing!", icon='ERROR')

            col = layout.column(align=True)
            col.prop(mod, "source_a")
            col.prop(mod, "source_b")

            layout.prop(mod, "operation", expand=True)
            layout.prop(mod, "live")

            row = layout.row(align=True)
            row.operator("mesh.trueform_boolean_refresh", icon='FILE_REFRESH')
            row.operator("mesh.trueform_boolean_remove", icon='X', text="Remove")
        else:
            # CREATE MODE: Show creation interface
            props = context.scene.trueform_boolean_create

            col = layout.column(align=True)
            col.prop(props, "target_a")
            col.prop(props, "target_b")

            layout.prop(props, "operation", expand=True)
            layout.operator("mesh.trueform_boolean_create", icon='MOD_BOOLEAN')


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

classes = (
    TrueformBooleanModifier,
    TrueformBooleanCreateProps,
    MESH_OT_trueform_boolean_create,
    MESH_OT_trueform_boolean_refresh,
    MESH_OT_trueform_boolean_remove,
    VIEW3D_PT_trueform_boolean,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.trueform_boolean = bpy.props.PointerProperty(type=TrueformBooleanModifier)
    bpy.types.Scene.trueform_boolean_create = bpy.props.PointerProperty(type=TrueformBooleanCreateProps)


def unregister():
    _remove_handlers()
    del bpy.types.Scene.trueform_boolean_create
    del bpy.types.Object.trueform_boolean
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
