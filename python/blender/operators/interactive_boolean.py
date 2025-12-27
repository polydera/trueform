"""
Interactive boolean operator with real-time intersection curve preview.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/xlabmedical/trueform
"""

import bpy
import numpy as np
from bpy.props import EnumProperty

import trueform as tf

from .. import meshes
from .. import convert

DEBUG = True
PREVIEW_MATERIAL_NAME = "Trueform_Preview_Orange"


class TRUEFORM_OT_interactive_boolean(bpy.types.Operator):
    """Interactive boolean with real-time intersection curve preview"""
    bl_idname = "trueform.interactive_boolean"
    bl_label = "Interactive Boolean"
    bl_options = {'REGISTER', 'UNDO'}

    operation: EnumProperty(
        name="Operation",
        items=[
            ('DIFFERENCE', "Difference", "A - B"),
            ('UNION', "Union", "A ∪ B"),
            ('INTERSECTION', "Intersection", "A ∩ B"),
        ],
        default='DIFFERENCE',
    )

    # Internal state
    _obj_a_name = None
    _obj_b_name = None
    _curves_name = None
    _last_matrix_a = None
    _last_matrix_b = None
    _handler = None

    def _log(self, message: str) -> None:
        if DEBUG:
            print(f"[TrueformInteractiveBoolean] {message}")

    @classmethod
    def poll(cls, context):
        """Check if exactly two mesh objects are selected"""
        selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
        return len(selected) == 2

    def invoke(self, context, event):
        """Start interactive boolean mode"""
        # Get the two selected meshes
        selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
        if len(selected) != 2:
            self.report({'ERROR'}, "Select exactly two mesh objects")
            return {'CANCELLED'}

        self._obj_a_name = selected[0].name
        self._obj_b_name = selected[1].name

        # Store initial transforms for change detection
        self._last_matrix_a = selected[0].matrix_world.copy()
        self._last_matrix_b = selected[1].matrix_world.copy()

        self._log(f"invoke: obj_a={self._obj_a_name}, obj_b={self._obj_b_name}")

        # Create initial curves preview
        self._update_curves(context)

        # Register depsgraph update handler
        self._handler = lambda scene, depsgraph: self._on_depsgraph_update(context, depsgraph)
        bpy.app.handlers.depsgraph_update_post.append(self._handler)

        # Start modal
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Boolean mode: D=Diff, U=Union, I=Intersect, Enter=Confirm, Esc=Cancel")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        """Handle events during interactive mode"""
        obj_a, obj_b, _curves_obj = self._resolve_objects()
        if obj_a is None or obj_b is None:
            self._log("modal: objects missing, cancelling")
            self._cancel(context)
            return {'CANCELLED'}

        # Operation switching
        if event.type == 'D' and event.value == 'PRESS':
            self.operation = 'DIFFERENCE'
            self._log("modal: operation=DIFFERENCE")
            self._update_curves(context)
            self.report({'INFO'}, "Operation: Difference")
            return {'RUNNING_MODAL'}

        elif event.type == 'U' and event.value == 'PRESS':
            self.operation = 'UNION'
            self._log("modal: operation=UNION")
            self._update_curves(context)
            self.report({'INFO'}, "Operation: Union")
            return {'RUNNING_MODAL'}

        elif event.type == 'I' and event.value == 'PRESS':
            self.operation = 'INTERSECTION'
            self._log("modal: operation=INTERSECTION")
            self._update_curves(context)
            self.report({'INFO'}, "Operation: Intersection")
            return {'RUNNING_MODAL'}

        # Confirm
        elif event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            self._log("modal: confirm")
            self._finish(context)
            return {'FINISHED'}

        # Cancel
        elif event.type == 'ESC' and event.value == 'PRESS':
            self._log("modal: cancel")
            self._cancel(context)
            return {'CANCELLED'}

        # Pass through other events (allows transform with G, etc.)
        return {'PASS_THROUGH'}

    def _on_depsgraph_update(self, context, depsgraph):
        """Called when depsgraph updates - check if our meshes moved"""
        obj_a, obj_b, _curves_obj = self._resolve_objects()
        if obj_a is None or obj_b is None:
            self._log("depsgraph: objects missing")
            return

        # Check if either transform changed
        matrix_a = obj_a.matrix_world
        matrix_b = obj_b.matrix_world

        changed = False
        if matrix_a != self._last_matrix_a:
            self._last_matrix_a = matrix_a.copy()
            changed = True
        if matrix_b != self._last_matrix_b:
            self._last_matrix_b = matrix_b.copy()
            changed = True

        if changed:
            self._log("depsgraph: transform changed, updating curves")
            self._update_curves(context)

    def _update_curves(self, context):
        """Update intersection curves preview"""
        try:
            obj_a, obj_b, curves_obj = self._resolve_objects()
            if obj_a is None or obj_b is None:
                self._log("update_curves: objects missing")
                return

            self._log(f"update_curves: obj_a={obj_a.name}, obj_b={obj_b.name}")

            # Get trueform meshes with current transforms
            mesh_a = meshes.get(obj_a)
            mesh_b = meshes.get(obj_b)

            # Compute intersection curves
            paths, points = tf.intersection_curves(mesh_a, mesh_b)
            self._log(f"update_curves: paths={len(paths)} points={len(points)}")

            # Update or create curves object
            if len(paths) > 0:
                if curves_obj is not None:
                    # Update existing curve data
                    old_curve = curves_obj.data
                    new_curve = convert.make_curves(paths, points, "Boolean_Preview_Curves")
                    curves_obj.data = new_curve
                    bpy.data.curves.remove(old_curve)
                    self._style_preview_curves(curves_obj)
                    self._log(f"update_curves: updated curves={curves_obj.name}")
                else:
                    # Create new curves object
                    curves_obj = convert.make_curves_object(paths, points, "Boolean_Preview_Curves")
                    self._curves_name = curves_obj.name
                    self._style_preview_curves(curves_obj)
                    self._log(f"update_curves: created curves={curves_obj.name}")
            else:
                # No intersection - remove curves if exists
                if curves_obj is not None:
                    bpy.data.objects.remove(curves_obj, do_unlink=True)
                    self._curves_name = None
                    self._log("update_curves: removed curves (no intersection)")

            # Force viewport update
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        except Exception as e:
            print(f"Error updating curves: {e}")
            self._log(f"update_curves exception: {e}")

    def _finish(self, context):
        """Confirm - compute boolean and create result"""
        # Remove handler
        self._cleanup_handler()

        # Remove preview curves
        self._remove_preview_curves()

        # Compute final boolean
        obj_a, obj_b, _curves_obj = self._resolve_objects()
        if obj_a is None or obj_b is None:
            self.report({'ERROR'}, "Meshes missing, boolean cancelled")
            self._log("finish: objects missing, cancelled")
            return

        mesh_a = meshes.get(obj_a)
        mesh_b = meshes.get(obj_b)

        if self.operation == 'DIFFERENCE':
            ((result_faces, result_points), _labels) = tf.boolean_difference(mesh_a, mesh_b)
        elif self.operation == 'UNION':
            ((result_faces, result_points), _labels) = tf.boolean_union(mesh_a, mesh_b)
        else:  # INTERSECTION
            ((result_faces, result_points), _labels) = tf.boolean_intersection(mesh_a, mesh_b)

        # Create result object
        result_obj = convert.make_mesh_object(result_faces, result_points, "Boolean_Result")

        # Select result
        bpy.ops.object.select_all(action='DESELECT')
        result_obj.select_set(True)
        context.view_layer.objects.active = result_obj

        self.report({'INFO'}, f"Boolean {self.operation.lower()} completed")
        self._log(f"finish: completed {self.operation.lower()}")

    def _cancel(self, context):
        """Cancel - cleanup without creating result"""
        # Remove handler
        self._cleanup_handler()

        # Remove preview curves
        self._remove_preview_curves()

        self.report({'INFO'}, "Boolean cancelled")
        self._log("cancel: cleaned up")

    def _cleanup_handler(self):
        """Remove depsgraph update handler"""
        if self._handler is not None:
            if self._handler in bpy.app.handlers.depsgraph_update_post:
                bpy.app.handlers.depsgraph_update_post.remove(self._handler)
            self._handler = None

    def _resolve_objects(self):
        """Get current objects by name, returning (obj_a, obj_b, curves_obj)."""
        obj_a = bpy.data.objects.get(self._obj_a_name) if self._obj_a_name else None
        obj_b = bpy.data.objects.get(self._obj_b_name) if self._obj_b_name else None
        curves_obj = bpy.data.objects.get(self._curves_name) if self._curves_name else None
        return obj_a, obj_b, curves_obj

    def _remove_preview_curves(self):
        """Remove the preview curve object if it still exists."""
        curves_obj = bpy.data.objects.get(self._curves_name) if self._curves_name else None
        if curves_obj is not None:
            bpy.data.objects.remove(curves_obj, do_unlink=True)
        self._curves_name = None

    def _style_preview_curves(self, curves_obj):
        """Apply a consistent preview style to curve objects."""
        curves_obj.data.bevel_depth = 0.02
        curves_obj.data.bevel_resolution = 3
        curves_obj.data.use_fill_caps = True
        curves_obj.color = (1.0, 0.55, 0.1, 1.0)  # Orange
        material = bpy.data.materials.get(PREVIEW_MATERIAL_NAME)
        if material is None:
            material = bpy.data.materials.new(name=PREVIEW_MATERIAL_NAME)
            material.use_nodes = True
            node_tree = material.node_tree
            if node_tree is not None:
                principled = node_tree.nodes.get("Principled BSDF")
                if principled is not None:
                    principled.inputs["Base Color"].default_value = (1.0, 0.55, 0.1, 1.0)
                    principled.inputs["Emission"].default_value = (1.0, 0.35, 0.05, 1.0)
                    principled.inputs["Emission Strength"].default_value = 0.6
            material.diffuse_color = (1.0, 0.55, 0.1, 1.0)
        if curves_obj.data.materials:
            curves_obj.data.materials[0] = material
        else:
            curves_obj.data.materials.append(material)
        curves_obj.active_material = material
