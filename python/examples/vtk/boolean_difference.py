"""
Interactive boolean difference example with VTK

Left viewport shows input meshes with intersection curves.
Right viewport shows the boolean difference result.
Operation time is averaged over the last 100 frames.
"""

import vtk
import numpy as np
import trueform as tf
import time

# Import utilities
from util import (
    BaseInteractor,
    RollingAverage,
    load_mesh,
    load_mesh_shared,
    numpy_to_polydata,
    curves_to_polydata,
    random_rotation_matrix,
    format_time_ms,
    create_text_actor,
    create_parser,
)


def set_matrix_from_numpy(vtk_matrix, numpy_matrix):
    """Copy numpy 4x4 matrix to VTK matrix"""
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, float(numpy_matrix[i, j]))
    vtk_matrix.Modified()


class BooleanDifferenceInteractor(BaseInteractor):
    """Interactive boolean difference visualization with dual viewports"""

    def __init__(self, mesh_data0, mesh_data1, result_polydata, curve_polydata, text_actor):
        """Initialize with two mesh data objects

        Args:
            mesh_data0: First MeshData (minuend)
            mesh_data1: Second MeshData (subtrahend)
            result_polydata: vtkPolyData for result mesh
            curve_polydata: vtkPolyData for intersection curves
            text_actor: vtkTextActor for performance display
        """
        super().__init__()

        self.mesh_data0 = mesh_data0
        self.mesh_data1 = mesh_data1
        self.result_polydata = result_polydata
        self.curve_polydata = curve_polydata
        self.text_actor = text_actor

        # Performance tracking
        self.times = RollingAverage(maxlen=100)

        # Enable mesh interaction (dragging)
        self.enable_mesh_interaction([mesh_data0, mesh_data1])

        # Add key press handler for randomization
        self.AddObserver("KeyPressEvent", self.on_key_press)

    def compute_boolean(self):
        """Compute boolean difference and update visualization"""
        start_time = time.perf_counter()

        # Get both meshes
        mesh0 = self.mesh_data0.mesh
        mesh1 = self.mesh_data1.mesh

        # Compute boolean difference with curves
        (result_faces, result_points), labels, (paths, curve_points) = (
            tf.boolean_difference(mesh0, mesh1, return_curves=True)
        )

        elapsed = time.perf_counter() - start_time
        self.times.add(elapsed)

        # Update result mesh (right viewport)
        if len(result_faces) > 0 and len(result_points) > 0:
            result_polydata = numpy_to_polydata(result_points, result_faces)
            self.result_polydata.ShallowCopy(result_polydata)
        else:
            # No result - clear visualization
            self.result_polydata.Initialize()

        self.result_polydata.Modified()

        # Update curves (left viewport)
        if len(paths) > 0 and len(curve_points) > 0:
            curve_polydata = curves_to_polydata(paths, curve_points)
            self.curve_polydata.ShallowCopy(curve_polydata)
        else:
            # No curves - clear visualization
            self.curve_polydata.Initialize()

        self.curve_polydata.Modified()

        self._update_text()

        # Only render if interactor is attached
        if self.GetInteractor() is not None:
            self.GetInteractor().Render()

    def randomize_orientations(self):
        """Randomize orientations of both meshes (rotate around their centers)"""
        for mesh_data in [self.mesh_data0, self.mesh_data1]:
            # Get current transformation as numpy array
            current_transform = mesh_data.mesh.transformation

            # Compute center of mesh in world space
            mesh_center = np.mean(mesh_data.mesh.points, axis=0)
            mesh_center_world = (current_transform @ np.append(mesh_center, 1.0))[:3]

            # Generate random rotation matrix (4x4)
            R = random_rotation_matrix(dtype=mesh_data.mesh.dtype)
            R_4x4 = np.eye(4, dtype=mesh_data.mesh.dtype)
            R_4x4[:3, :3] = R

            # Rotate around mesh center: T(center) @ R @ T(-center) @ current
            translate_to_origin = np.eye(4, dtype=mesh_data.mesh.dtype)
            translate_to_origin[:3, 3] = -mesh_center_world

            translate_back = np.eye(4, dtype=mesh_data.mesh.dtype)
            translate_back[:3, 3] = mesh_center_world

            new_transform = translate_back @ R_4x4 @ translate_to_origin @ current_transform

            # Update matrices
            set_matrix_from_numpy(mesh_data.matrix, new_transform)
            mesh_data.mesh.transformation = new_transform

        # Recompute boolean
        self.compute_boolean()

    def _update_text(self):
        """Update text display with performance info"""
        avg_time = self.times.get_average()
        time_str = format_time_ms(avg_time)
        self.text_actor.SetInput(f"Boolean difference time: {time_str}")

    def on_mesh_dragged(self, mesh_data, delta):
        """Called when a mesh is dragged - recompute boolean

        Args:
            mesh_data: The MeshData object that was dragged
            delta: Movement delta (not used)
        """
        self.compute_boolean()

    def on_key_press(self, obj, event):
        """Handle key press events

        Args:
            obj: vtkRenderWindowInteractor
            event: Event name
        """
        key = self.GetInteractor().GetKeySym()

        if key == "n":
            self.randomize_orientations()
        else:
            # Pass to base class for camera controls
            vtk.vtkInteractorStyleTrackballCamera.OnKeyPress(self)


def main():
    # Parse command line arguments
    parser = create_parser("Interactive boolean difference", mesh_args=2)
    parser.epilog = """
Controls:
  N            Randomize mesh orientations
  Mouse drag   Move meshes / rotate camera
"""
    args = parser.parse_args()
    mesh_file1 = args.mesh1
    mesh_file2 = args.mesh2

    # Load first mesh (use dynamic=True to test OffsetBlockedArray)
    mesh_data0 = load_mesh(mesh_file1, (0.0, 0.0, 0.0), target_radius=10.0, random_rotation=True, dynamic=True)

    # Build structures once on the first mesh
    mesh_data0.mesh.build_tree()
    mesh_data0.mesh.build_face_membership()
    mesh_data0.mesh.build_manifold_edge_link()

    if mesh_file2 is None:
        # Same file - use shared_view to share geometry and structures
        mesh_data1 = load_mesh_shared(mesh_data0, (15.0, 0.0, 0.0), random_rotation=True)
    else:
        # Different files - load separately
        mesh_data1 = load_mesh(mesh_file2, (15.0, 0.0, 0.0), target_radius=10.0, random_rotation=True, dynamic=True)
        mesh_data1.mesh.build_tree()
        mesh_data1.mesh.build_face_membership()
        mesh_data1.mesh.build_manifold_edge_link()

    # Create renderers (left + right viewports, plus text strip at bottom)
    renderer_left = vtk.vtkRenderer()
    renderer_right = vtk.vtkRenderer()
    renderer_text = vtk.vtkRenderer()

    # Viewports: two side-by-side on top (88% height), text bar bottom (12%)
    renderer_left.SetViewport(0.0, 0.12, 0.5, 1.0)
    renderer_right.SetViewport(0.5, 0.12, 1.0, 1.0)
    renderer_text.SetViewport(0.0, 0.0, 1.0, 0.12)
    renderer_text.InteractiveOff()

    # Backgrounds (dark blue/gray matching isobands)
    renderer_left.SetBackground(27.0 / 255.0, 43.0 / 255.0, 52.0 / 255.0)
    renderer_right.SetBackground(27.0 / 255.0, 43.0 / 255.0, 52.0 / 255.0)
    renderer_text.SetBackground(0.090, 0.143, 0.173)

    # === LEFT VIEWPORT: Both input meshes + intersection curves ===

    # Add both mesh actors to left viewport
    renderer_left.AddActor(mesh_data0.actor)
    renderer_left.AddActor(mesh_data1.actor)

    # Setup curve actor on left (initially empty)
    curve_poly = vtk.vtkPolyData()
    curve_poly.Initialize()
    curve_mapper = vtk.vtkPolyDataMapper()
    curve_mapper.SetInputData(curve_poly)
    curve_actor = vtk.vtkActor()
    curve_actor.SetMapper(curve_mapper)

    # Render lines as tubes (GPU-accelerated)
    curve_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red curves
    curve_actor.GetProperty().SetRenderLinesAsTubes(True)
    curve_actor.GetProperty().SetLineWidth(8.0)  # Tube width in pixels

    renderer_left.AddActor(curve_actor)

    # === RIGHT VIEWPORT: Boolean difference result mesh ===

    # Setup result mesh actor (initially empty)
    result_poly = vtk.vtkPolyData()
    result_poly.Initialize()
    mapper_right = vtk.vtkPolyDataMapper()
    mapper_right.SetInputData(result_poly)
    actor_right = vtk.vtkActor()
    actor_right.SetMapper(mapper_right)
    actor_right.GetProperty().SetColor(0.8, 0.8, 0.8)
    renderer_right.AddActor(actor_right)

    # === TEXT STRIP ===

    text_time = create_text_actor(
        "Boolean difference time: 0 ms",
        font_size=38,
        position=(0.03, 0.50),
        justification='left'
    )
    renderer_text.AddViewProp(text_time)

    # === RENDER WINDOW AND INTERACTOR ===

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer_left)
    render_window.AddRenderer(renderer_right)
    render_window.AddRenderer(renderer_text)
    render_window.SetSize(1200, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Setup custom interactor style
    style = BooleanDifferenceInteractor(mesh_data0, mesh_data1, result_poly, curve_poly, text_time)
    interactor.SetInteractorStyle(style)

    # Share camera between left and right viewports
    renderer_right.SetActiveCamera(renderer_left.GetActiveCamera())

    # Reset cameras
    renderer_left.ResetCamera()
    renderer_right.ResetCamera()

    # Compute initial boolean
    style.compute_boolean()

    # Start
    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
