"""
Intersection Curves modifier tool for Trueform Blender addon.

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

def _is_trueform_curves_result(obj) -> bool:
    """Check if object is a Trueform curves result."""
    if not obj or obj.type != 'CURVE':
        return False
    mod = getattr(obj, 'trueform_curves', None)
    return mod is not None and mod.source_a is not None


def _mesh_poll(self, obj):
    """Poll function for source selection - must be mesh."""
    return obj.type == 'MESH'


def _get_live_results():
    """Get all curves result objects with live=True."""
    results = []
    for obj in bpy.data.objects:
        if obj.type != 'CURVE':
            continue
        mod = getattr(obj, 'trueform_curves', None)
        if mod and mod.live and mod.source_a and mod.source_b:
            results.append(obj)
    return results


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

def _style_curves(curves_obj, obj_a, obj_b):
    """Apply default styling to curves object."""
    if obj_a and obj_b:
        size = min(max(obj_a.dimensions), max(obj_b.dimensions))
        curves_obj.data.bevel_depth = size * 0.005
    else:
        curves_obj.data.bevel_depth = 0.02
    curves_obj.data.bevel_resolution = 3
    curves_obj.data.use_fill_caps = True
    core.apply_material(curves_obj, core.get_teal_material())


# -----------------------------------------------------------------------------
# Update Logic
# -----------------------------------------------------------------------------

def _update_result(result_obj):
    """Update a single curves result object from its sources."""
    mod = result_obj.trueform_curves
    if not mod or not mod.source_a or not mod.source_b:
        return
    if mod.source_a == mod.source_b:
        return

    tf, tfb = core.get_tf_libs()
    if not tf or not tfb:
        return

    try:
        mesh_a = tfb.scene.get(mod.source_a)
        mesh_b = tfb.scene.get(mod.source_b)
        paths, points = tf.intersection_curves(mesh_a, mesh_b)

        if paths:
            tfb.convert.update_curves(paths, points, result_obj)
        else:
            # No intersection - clear splines
            result_obj.data.splines.clear()

        core.tag_view3d_redraw(bpy.context)
    except Exception as e:
        print(f"Trueform Curves Error on '{result_obj.name}': {e}")


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
        mod = result_obj.trueform_curves
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
    """Called when source_a or source_b changes on a result."""
    obj = context.active_object
    if obj and _is_trueform_curves_result(obj) and self.live:
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

class TrueformCurvesModifier(bpy.types.PropertyGroup):
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
    live: bpy.props.BoolProperty(
        name="Live",
        default=True,
        update=_on_live_toggle
    )


class TrueformCurvesCreateProps(bpy.types.PropertyGroup):
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


# -----------------------------------------------------------------------------
# Operators
# -----------------------------------------------------------------------------

class MESH_OT_trueform_curves_create(bpy.types.Operator):
    """Create intersection curves from two meshes"""
    bl_idname = "mesh.trueform_curves_create"
    bl_label = "Create Curves"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.trueform_curves_create

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
            mesh_a = tfb.scene.get(props.target_a)
            mesh_b = tfb.scene.get(props.target_b)
            paths, points = tf.intersection_curves(mesh_a, mesh_b)

            if not paths:
                self.report({'INFO'}, "No intersection found")
                return {'CANCELLED'}

            # Create curves object
            curves_obj = tfb.convert.to_blender_curves(
                paths, points,
                f"TF_Curves_{props.target_a.name}"
            )
            _style_curves(curves_obj, props.target_a, props.target_b)

            # Store modifier data on result
            mod = curves_obj.trueform_curves
            mod.source_a = props.target_a
            mod.source_b = props.target_b
            mod.live = True

            # Register handlers
            _ensure_handlers()

            # Select result
            bpy.ops.object.select_all(action='DESELECT')
            curves_obj.select_set(True)
            context.view_layer.objects.active = curves_obj

            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}


class MESH_OT_trueform_curves_refresh(bpy.types.Operator):
    """Force refresh the curves result"""
    bl_idname = "mesh.trueform_curves_refresh"
    bl_label = "Refresh"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and _is_trueform_curves_result(context.active_object)

    def execute(self, context):
        _update_result(context.active_object)
        return {'FINISHED'}


class MESH_OT_trueform_curves_remove(bpy.types.Operator):
    """Remove Trueform curves modifier, keeping result as static curves"""
    bl_idname = "mesh.trueform_curves_remove"
    bl_label = "Remove Modifier"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and _is_trueform_curves_result(context.active_object)

    def execute(self, context):
        obj = context.active_object
        mod = obj.trueform_curves

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

class VIEW3D_PT_trueform_curves(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Trueform'
    bl_label = "Intersection Curves"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        if obj and _is_trueform_curves_result(obj):
            # MODIFIER MODE: Show result's settings
            mod = obj.trueform_curves

            # Warning if sources missing
            if not mod.source_a or not mod.source_b:
                layout.label(text="Source missing!", icon='ERROR')

            col = layout.column(align=True)
            col.prop(mod, "source_a")
            col.prop(mod, "source_b")

            layout.prop(mod, "live")

            row = layout.row(align=True)
            row.operator("mesh.trueform_curves_refresh", icon='FILE_REFRESH')
            row.operator("mesh.trueform_curves_remove", icon='X', text="Remove")
        else:
            # CREATE MODE: Show creation interface
            props = context.scene.trueform_curves_create

            col = layout.column(align=True)
            col.prop(props, "target_a")
            col.prop(props, "target_b")

            layout.operator("mesh.trueform_curves_create", icon='OUTLINER_DATA_CURVE')


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

classes = (
    TrueformCurvesModifier,
    TrueformCurvesCreateProps,
    MESH_OT_trueform_curves_create,
    MESH_OT_trueform_curves_refresh,
    MESH_OT_trueform_curves_remove,
    VIEW3D_PT_trueform_curves,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.trueform_curves = bpy.props.PointerProperty(type=TrueformCurvesModifier)
    bpy.types.Scene.trueform_curves_create = bpy.props.PointerProperty(type=TrueformCurvesCreateProps)


def unregister():
    _remove_handlers()
    del bpy.types.Scene.trueform_curves_create
    del bpy.types.Object.trueform_curves
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
