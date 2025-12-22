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
def get_tf_libs():
    if LIB_PATH is not None and LIB_PATH not in sys.path:
        sys.path.insert(0, LIB_PATH)
    try:
        import trueform as tf
        import trueform.blender as tfb
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

# --- PROPERTY GROUP ---
class TrueformProperties(bpy.types.PropertyGroup):
    target_a: bpy.props.PointerProperty(
        name="Mesh A",
        description="First mesh for boolean",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH'
    )
    target_b: bpy.props.PointerProperty(
        name="Mesh B",
        description="Second mesh for boolean",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH'
    )
    operation: bpy.props.EnumProperty(
        name="Operation",
        description="Boolean operation to apply",
        items=[
            ('DIFFERENCE', "Difference", "A - B"),
            ('UNION', "Union", "A + B"),
            ('INTERSECTION', "Intersection", "A & B"),
        ],
        default='INTERSECTION'
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

        start_time = time.time()

        try:
            # Execute boolean
            if props.operation == 'DIFFERENCE':
                result_obj = tfb.functions.boolean_difference(obj_a, obj_b)
            elif props.operation == 'UNION':
                result_obj = tfb.functions.boolean_union(obj_a, obj_b)
            else:
                result_obj = tfb.functions.boolean_intersection(obj_a, obj_b)

            # Name and organize
            result_obj.name = f"TFB_{obj_a.name}_{obj_b.name}"

            if props.hide_inputs:
                obj_a.hide_set(True)
                obj_b.hide_set(True)

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
        box.prop(props, "hide_inputs")

        layout.separator()

        row = layout.row(align=True)
        row.scale_y = 1.4
        row.operator("mesh.trueform_boolean", text="Apply Boolean", icon='MOD_BOOLEAN')
        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.4
        row.operator("mesh.trueform_interactive_boolean", text="Interactive Preview", icon='MOD_BOOLEAN')

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
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.trueform_tools

if __name__ == "__main__":
    register()
