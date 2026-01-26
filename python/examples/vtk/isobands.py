"""
Interactive isoband extraction example with VTK

The scalar field is computed as signed distance from a random plane.
Multiple isobands are extracted simultaneously with every other band selected.
Boundary curves between bands are visualized as red tubes.
Extraction time is averaged over the last 100 frames.
"""

import vtk
import numpy as np
import trueform as tf

# Import utilities
from util import (
    MeshData,
    numpy_to_polydata,
    compute_centering_and_scaling_transform,
    curves_to_polydata,
    BaseInteractor,
    RollingAverage,
    format_time_ms,
    create_text_actor,
    create_parser,
)

# Set to True to test dynamic mesh (OffsetBlockedArray)
USE_DYNAMIC_MESH = True


class IsobandInteractor(BaseInteractor):
    """Interactor for isoband visualization with scrolling and randomization"""

    def __init__(self, mesh_data, isobands_polydata, curve_polydata, text_actor):
        super().__init__()

        # Keep mesh data (wraps trueform mesh + actor)
        self.mesh_data = mesh_data
        self.mesh = mesh_data.mesh  # Convenience reference
        self.isobands_polydata = isobands_polydata
        self.curve_polydata = curve_polydata
        self.text_actor = text_actor

        # Scalar field data
        self.scalars = None
        self.distance = 0.0  # Offset for isoband levels
        self.min_d = 0.0     # Min scalar value
        self.max_d = 0.0     # Max scalar value
        self.spacing = 0.1   # Will be computed in reset_plane

        # Timing
        self.times = RollingAverage(maxlen=100)

        # Initialize random plane (but don't compute bands yet - interactor not attached)
        self.reset_plane()

        # Add custom event handlers for scrolling and keys
        self.AddObserver("KeyPressEvent", self.on_key_press)
        self.AddObserver("MouseWheelForwardEvent", self.on_mouse_wheel_forward)
        self.AddObserver("MouseWheelBackwardEvent", self.on_mouse_wheel_backward)

    def reset_plane(self):
        """Create a new random plane and compute scalar field"""
        # Create random plane through a random point
        random_normal = np.random.randn(3).astype(self.mesh.dtype)
        random_normal /= np.linalg.norm(random_normal)
        random_point = self.mesh.points[np.random.randint(0, len(self.mesh.points))]

        plane = tf.Plane.from_point_normal(random_point, random_normal)

        # Compute scalar field (signed distance to plane)
        self.scalars = tf.distance_field(self.mesh.points, plane)
        self.distance = 0.0

        # Cache min/max for adaptive level spacing
        self.min_d = float(np.min(self.scalars))
        self.max_d = float(np.max(self.scalars))
        self.spacing = (self.max_d - self.min_d) / 10.0  # N = 10 bands

    def compute_bands(self):
        """Extract isobands with curves and update visualization"""
        import time as time_module

        # Compute adaptive threshold levels (same algorithm as isobands.cpp)
        N = 10  # Number of bands

        # Use fmod to wrap offset within spacing (infinite scrolling)
        wrapped_offset = self.distance % self.spacing
        if wrapped_offset < 0:
            wrapped_offset += self.spacing

        # Generate evenly spaced cut values offset by wrapped_offset
        # Extend below and above the range so wrapping is seamless
        cutvalues = np.array(
            [self.min_d + wrapped_offset + i * self.spacing for i in range(-1, N + 2)],
            dtype=self.mesh.dtype
        )

        # Select every other band (alternating pattern)
        # Now we have N+2 bands (indices 0 to N+1)
        parity = int(np.floor(self.distance / self.spacing)) & 1
        selected_bands = np.array([i for i in range(N + 2) if (i & 1) == parity], dtype=np.int32)

        # Extract isobands with curves and timing
        start_time = time_module.perf_counter()
        (band_faces, band_points), labels, (paths, curve_points) = tf.isobands(
            self.mesh, self.scalars, cutvalues, selected_bands, return_curves=True
        )
        elapsed = time_module.perf_counter() - start_time

        # Update timing
        self.times.add(elapsed)
        avg_time = self.times.get_average()
        self.text_actor.SetInput(f"Isobands time: {format_time_ms(avg_time)}")

        # Convert isobands mesh to polydata
        if len(band_faces) > 0 and len(band_points) > 0:
            poly = numpy_to_polydata(band_points, band_faces)
            self.isobands_polydata.ShallowCopy(poly)
        else:
            # No bands - clear visualization
            self.isobands_polydata.Initialize()

        self.isobands_polydata.Modified()

        # Convert curves to polydata
        if len(paths) > 0 and len(curve_points) > 0:
            curve_poly = curves_to_polydata(paths, curve_points)
            self.curve_polydata.ShallowCopy(curve_poly)
        else:
            # No curves - clear visualization
            self.curve_polydata.Initialize()

        self.curve_polydata.Modified()

        # Only render if interactor is attached
        if self.GetInteractor() is not None:
            self.GetInteractor().Render()

    def on_key_press(self, obj, event):
        """Handle key press events"""
        key = self.GetInteractor().GetKeySym()
        if key == "n":
            # Randomize plane
            self.reset_plane()
            self.compute_bands()
        else:
            # Pass to base class for camera controls
            vtk.vtkInteractorStyleTrackballCamera.OnKeyPress(self)

    def on_mouse_wheel_forward(self, obj, event):
        """Handle mouse wheel forward"""
        if self.GetInteractor().GetControlKey():
            # Move isobands forward (same as C++: spacing * 0.1)
            self.distance += self.spacing * 0.1
            self.compute_bands()
        else:
            # Pass to base class for zoom
            vtk.vtkInteractorStyleTrackballCamera.OnMouseWheelForward(self)

    def on_mouse_wheel_backward(self, obj, event):
        """Handle mouse wheel backward"""
        if self.GetInteractor().GetControlKey():
            # Move isobands backward (same as C++: spacing * 0.1)
            self.distance -= self.spacing * 0.1
            self.compute_bands()
        else:
            # Pass to base class for zoom
            vtk.vtkInteractorStyleTrackballCamera.OnMouseWheelBackward(self)


def main():
    # Parse command line arguments
    parser = create_parser("Interactive isoband extraction", mesh_args=1)
    parser.epilog = """
Controls:
  N              Randomize cutting plane
  Ctrl + Scroll  Move isoband levels
  Mouse drag     Rotate camera
"""
    args = parser.parse_args()
    mesh_file = args.mesh

    # Load mesh
    faces, points = tf.read_stl(mesh_file)

    # Optionally convert to dynamic mesh
    if USE_DYNAMIC_MESH:
        faces = tf.as_offset_blocked(faces)

    # Center and scale mesh
    transform = compute_centering_and_scaling_transform(points, target_radius=10.0)

    # Apply transformation
    points_homogeneous = np.hstack([points, np.ones((len(points), 1), dtype=points.dtype)])
    points_transformed = (transform @ points_homogeneous.T).T[:, :3]

    # Create trueform mesh FIRST (this is the primary data structure)
    mesh = tf.Mesh(faces, points_transformed)

    # Create VTK polydata for visualization (just a view)
    poly = numpy_to_polydata(points_transformed, faces)

    # Create renderers (left + right viewports, plus text strip at bottom)
    renderer_left = vtk.vtkRenderer()
    renderer_right = vtk.vtkRenderer()
    renderer_text = vtk.vtkRenderer()

    # Viewports: two side-by-side on top (88% height), text bar bottom (12%)
    renderer_left.SetViewport(0.0, 0.12, 0.5, 1.0)
    renderer_right.SetViewport(0.5, 0.12, 1.0, 1.0)
    renderer_text.SetViewport(0.0, 0.0, 1.0, 0.12)
    renderer_text.InteractiveOff()

    # Backgrounds (dark blue/gray matching C++)
    renderer_left.SetBackground(27.0 / 255.0, 43.0 / 255.0, 52.0 / 255.0)
    renderer_right.SetBackground(27.0 / 255.0, 43.0 / 255.0, 52.0 / 255.0)
    renderer_text.SetBackground(0.090, 0.143, 0.173)

    # === LEFT VIEWPORT: Original mesh with curves ===

    # Setup original mesh actor
    mapper_left = vtk.vtkPolyDataMapper()
    mapper_left.SetInputData(poly)
    actor_left = vtk.vtkActor()
    actor_left.SetMapper(mapper_left)
    actor_left.GetProperty().SetColor(0.8, 0.8, 0.8)
    renderer_left.AddActor(actor_left)

    # Create VTK matrix for transformations
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    actor_left.SetUserMatrix(matrix)

    # Wrap in MeshData for interaction
    mesh_data = MeshData(mesh, actor_left, matrix)

    # Setup curve actor on left (initially empty)
    curve_poly = vtk.vtkPolyData()
    curve_poly.Initialize()
    curve_mapper = vtk.vtkPolyDataMapper()
    curve_mapper.SetInputData(curve_poly)
    curve_actor = vtk.vtkActor()
    curve_actor.SetMapper(curve_mapper)

    # Render lines as tubes (GPU-accelerated)
    curve_actor.GetProperty().SetColor(1.0, 0.1, 0.1)  # Red curves
    curve_actor.GetProperty().SetRenderLinesAsTubes(True)
    curve_actor.GetProperty().SetLineWidth(8.0)  # Tube width in pixels

    renderer_left.AddActor(curve_actor)

    # === RIGHT VIEWPORT: Isobands mesh ===

    # Setup isobands mesh actor (initially empty)
    isobands_poly = vtk.vtkPolyData()
    isobands_poly.Initialize()
    mapper_right = vtk.vtkPolyDataMapper()
    mapper_right.SetInputData(isobands_poly)
    actor_right = vtk.vtkActor()
    actor_right.SetMapper(mapper_right)
    actor_right.GetProperty().SetColor(0.8, 0.8, 0.8)
    renderer_right.AddActor(actor_right)

    # === TEXT STRIP ===

    text_time = create_text_actor(
        "Isobands time: 0 ms",
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
    style = IsobandInteractor(mesh_data, isobands_poly, curve_poly, text_time)
    interactor.SetInteractorStyle(style)

    # Share camera between left and right viewports
    renderer_right.SetActiveCamera(renderer_left.GetActiveCamera())

    # Reset cameras
    renderer_left.ResetCamera()
    renderer_right.ResetCamera()

    # Compute initial bands (now that interactor is attached)
    style.compute_bands()

    # Start
    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
