"""
Interactive intersection curves extraction example with VTK

Multiple intersection curves may be extracted depending on mesh topology.
Extraction time is averaged over the last 100 frames.
"""

import vtk
import numpy as np
import trueform as tf
import time

# Import utilities
from util import (
    load_mesh,
    random_rotation_matrix,
    curves_to_polydata,
    BaseInteractor,
    RollingAverage,
    format_time_ms,
    create_text_actor,
    create_renderer_with_text_strip,
    create_parser,
)


def set_matrix_from_numpy(vtk_matrix, numpy_matrix):
    """Copy numpy 4x4 matrix to VTK matrix"""
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, float(numpy_matrix[i, j]))
    vtk_matrix.Modified()


class IntersectionCurvesInteractor(BaseInteractor):
    """Interactor for intersection curves visualization with dragging and randomization"""

    def __init__(self, mesh_data0, mesh_data1, curve_polydata, text_actor):
        super().__init__()

        # Keep both mesh data objects
        self.mesh_data0 = mesh_data0
        self.mesh_data1 = mesh_data1
        self.curve_polydata = curve_polydata
        self.text_actor = text_actor

        # Timing
        self.times = RollingAverage(maxlen=100)

        # Enable mesh interaction (handles all picking/dragging automatically!)
        self.enable_mesh_interaction([mesh_data0, mesh_data1])

        # Add key press handler for randomization
        self.AddObserver("KeyPressEvent", self.on_key_press)

    def compute_curves(self):
        """Compute intersection curves between the two meshes"""
        # Extract intersection curves with timing
        start_time = time.perf_counter()
        paths, curve_points = tf.intersection_curves(self.mesh_data0.mesh, self.mesh_data1.mesh)
        elapsed = time.perf_counter() - start_time

        # Update timing
        self.times.add(elapsed)
        avg_time = self.times.get_average()
        self.text_actor.SetInput(f"Intersection curve time: {format_time_ms(avg_time)}")

        # Convert curves to polydata
        if len(paths) > 0 and len(curve_points) > 0:
            poly = curves_to_polydata(paths, curve_points)
            self.curve_polydata.ShallowCopy(poly)
        else:
            # No curves - clear visualization
            self.curve_polydata.Initialize()

        self.curve_polydata.Modified()

    def randomize_orientations(self):
        """Randomize orientations of both meshes (rotate around their centers)"""
        for mesh_data in [self.mesh_data0, self.mesh_data1]:
            # Get current transformation as numpy array
            current_transform = mesh_data.mesh.transformation

            # Compute center of mesh in world space (transform the mean of points)
            mesh_center = np.mean(mesh_data.mesh.points, axis=0)  # Center in local space
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

            # Update VTK matrix
            set_matrix_from_numpy(mesh_data.matrix, new_transform)

            # Update trueform mesh transformation
            mesh_data.mesh.transformation = new_transform

    def on_key_press(self, obj, event):
        """Handle key press events"""
        key = self.GetInteractor().GetKeySym()
        if key == "n":
            # Randomize orientations
            self.randomize_orientations()
            self.compute_curves()
            self.GetInteractor().Render()
        else:
            # Pass to base class for camera controls
            vtk.vtkInteractorStyleTrackballCamera.OnKeyPress(self)

    # Override hooks from BaseInteractor
    def on_mesh_dragged(self, mesh_data, delta):
        """Called when a mesh is dragged - recompute curves"""
        self.compute_curves()


def main():
    # Parse command line arguments
    parser = create_parser("Interactive intersection curves extraction", mesh_args=2)
    parser.epilog = """
Controls:
  N            Randomize mesh orientations
  Mouse drag   Move meshes / rotate camera
"""
    args = parser.parse_args()
    mesh_file1 = args.mesh1
    mesh_file2 = args.mesh2 if args.mesh2 else args.mesh1

    # Load meshes with random rotations at specified positions
    # Use dynamic=True to test OffsetBlockedArray
    mesh_data0 = load_mesh(mesh_file1, (0.0, 0.0, 0.0), target_radius=10.0, random_rotation=True, dynamic=True)
    mesh_data1 = load_mesh(mesh_file2, (15.0, 0.0, 0.0), target_radius=10.0, random_rotation=True, dynamic=True)

    # Build trees for ray casting
    mesh_data0.mesh.build_tree()
    mesh_data1.mesh.build_tree()
    mesh_data0.mesh.build_face_membership()
    mesh_data1.mesh.build_face_membership()
    mesh_data0.mesh.build_manifold_edge_link()
    mesh_data1.mesh.build_manifold_edge_link()

    # Create renderers (main + text strip)
    renderer, renderer_text = create_renderer_with_text_strip()

    # Add mesh actors to main renderer
    renderer.AddActor(mesh_data0.actor)
    renderer.AddActor(mesh_data1.actor)

    # Setup curve actor (initially empty)
    curve_poly = vtk.vtkPolyData()
    curve_poly.Initialize()
    curve_mapper = vtk.vtkPolyDataMapper()
    curve_mapper.SetInputData(curve_poly)
    curve_actor = vtk.vtkActor()
    curve_actor.SetMapper(curve_mapper)

    # Render lines as tubes (GPU-accelerated, no geometry generation needed)
    curve_actor.GetProperty().SetColor(1.0, 0.1, 0.1)  # Red curves
    curve_actor.GetProperty().SetRenderLinesAsTubes(True)
    curve_actor.GetProperty().SetLineWidth(8.0)  # Tube width in pixels

    renderer.AddActor(curve_actor)

    # Create text actor for timing
    text_time = create_text_actor(
        "Intersection curve time: 0 ms",
        font_size=38,
        position=(0.03, 0.50),
        justification='left'
    )
    renderer_text.AddViewProp(text_time)

    # Setup render window and interactor
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.AddRenderer(renderer_text)
    render_window.SetSize(800, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Setup custom interactor style
    style = IntersectionCurvesInteractor(mesh_data0, mesh_data1, curve_poly, text_time)
    interactor.SetInteractorStyle(style)

    # Reset camera
    renderer.ResetCamera()

    # Compute initial curves
    style.compute_curves()

    # Start
    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
