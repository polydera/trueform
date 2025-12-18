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
    _obj_a = None
    _obj_b = None
    _curves_obj = None
    _last_matrix_a = None
    _last_matrix_b = None
    _handler = None

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

        self._obj_a = selected[0]
        self._obj_b = selected[1]

        # Store initial transforms for change detection
        self._last_matrix_a = self._obj_a.matrix_world.copy()
        self._last_matrix_b = self._obj_b.matrix_world.copy()

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
        # Operation switching
        if event.type == 'D' and event.value == 'PRESS':
            self.operation = 'DIFFERENCE'
            self._update_curves(context)
            self.report({'INFO'}, "Operation: Difference")
            return {'RUNNING_MODAL'}

        elif event.type == 'U' and event.value == 'PRESS':
            self.operation = 'UNION'
            self._update_curves(context)
            self.report({'INFO'}, "Operation: Union")
            return {'RUNNING_MODAL'}

        elif event.type == 'I' and event.value == 'PRESS':
            self.operation = 'INTERSECTION'
            self._update_curves(context)
            self.report({'INFO'}, "Operation: Intersection")
            return {'RUNNING_MODAL'}

        # Confirm
        elif event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            self._finish(context)
            return {'FINISHED'}

        # Cancel
        elif event.type == 'ESC' and event.value == 'PRESS':
            self._cancel(context)
            return {'CANCELLED'}

        # Pass through other events (allows transform with G, etc.)
        return {'PASS_THROUGH'}

    def _on_depsgraph_update(self, context, depsgraph):
        """Called when depsgraph updates - check if our meshes moved"""
        if self._obj_a is None or self._obj_b is None:
            return

        # Check if either transform changed
        matrix_a = self._obj_a.matrix_world
        matrix_b = self._obj_b.matrix_world

        changed = False
        if matrix_a != self._last_matrix_a:
            self._last_matrix_a = matrix_a.copy()
            changed = True
        if matrix_b != self._last_matrix_b:
            self._last_matrix_b = matrix_b.copy()
            changed = True

        if changed:
            self._update_curves(context)

    def _update_curves(self, context):
        """Update intersection curves preview"""
        try:
            # Get trueform meshes with current transforms
            mesh_a = meshes.get(self._obj_a)
            mesh_b = meshes.get(self._obj_b)

            # Compute intersection curves
            paths, points = tf.intersection_curves(mesh_a, mesh_b)

            # Update or create curves object
            if len(paths) > 0:
                if self._curves_obj is not None:
                    # Update existing curve data
                    old_curve = self._curves_obj.data
                    new_curve = convert.make_curves(paths, points, "Boolean_Preview_Curves")
                    self._curves_obj.data = new_curve
                    bpy.data.curves.remove(old_curve)
                else:
                    # Create new curves object
                    self._curves_obj = convert.make_curves_object(paths, points, "Boolean_Preview_Curves")
                    # Style the curves
                    self._curves_obj.data.bevel_depth = 0.02
                    self._curves_obj.color = (1.0, 0.2, 0.2, 1.0)  # Red
            else:
                # No intersection - remove curves if exists
                if self._curves_obj is not None:
                    bpy.data.objects.remove(self._curves_obj, do_unlink=True)
                    self._curves_obj = None

            # Force viewport update
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        except Exception as e:
            print(f"Error updating curves: {e}")

    def _finish(self, context):
        """Confirm - compute boolean and create result"""
        # Remove handler
        self._cleanup_handler()

        # Remove preview curves
        if self._curves_obj is not None:
            bpy.data.objects.remove(self._curves_obj, do_unlink=True)
            self._curves_obj = None

        # Compute final boolean
        mesh_a = meshes.get(self._obj_a)
        mesh_b = meshes.get(self._obj_b)

        if self.operation == 'DIFFERENCE':
            result_faces, result_points = tf.boolean_difference(mesh_a, mesh_b)
        elif self.operation == 'UNION':
            result_faces, result_points = tf.boolean_union(mesh_a, mesh_b)
        else:  # INTERSECTION
            result_faces, result_points = tf.boolean_intersection(mesh_a, mesh_b)

        # Create result object
        result_obj = convert.make_mesh_object(result_faces, result_points, "Boolean_Result")

        # Select result
        bpy.ops.object.select_all(action='DESELECT')
        result_obj.select_set(True)
        context.view_layer.objects.active = result_obj

        self.report({'INFO'}, f"Boolean {self.operation.lower()} completed")

    def _cancel(self, context):
        """Cancel - cleanup without creating result"""
        # Remove handler
        self._cleanup_handler()

        # Remove preview curves
        if self._curves_obj is not None:
            bpy.data.objects.remove(self._curves_obj, do_unlink=True)
            self._curves_obj = None

        self.report({'INFO'}, "Boolean cancelled")

    def _cleanup_handler(self):
        """Remove depsgraph update handler"""
        if self._handler is not None:
            if self._handler in bpy.app.handlers.depsgraph_update_post:
                bpy.app.handlers.depsgraph_update_post.remove(self._handler)
            self._handler = None
