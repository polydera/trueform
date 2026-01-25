"""
Interactive isocontour extraction example with VTK

The scalar field is computed as signed distance from a random plane.
Multiple isocontour levels are extracted simultaneously and visualized as tubes.
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
    create_renderer_with_text_strip,
    create_parser,
)

# Set to True to test dynamic mesh (OffsetBlockedArray)
USE_DYNAMIC_MESH = True


class IsocontourInteractor(BaseInteractor):
    """Interactor for isocontour visualization with scrolling and randomization"""

    def __init__(self, mesh_data, curve_polydata, text_actor):
        super().__init__()

        # Keep mesh data (wraps trueform mesh + actor)
        self.mesh_data = mesh_data
        self.mesh = mesh_data.mesh  # Convenience reference
        self.curve_polydata = curve_polydata
        self.text_actor = text_actor

        # Scalar field data
        self.scalars = None
        self.distance = 0.0  # Offset for isocontour levels
        self.min_d = 0.0     # Min scalar value
        self.max_d = 0.0     # Max scalar value

        # Timing
        self.times = RollingAverage(maxlen=100)

        # Initialize random plane (but don't compute curves yet - interactor not attached)
        self.reset_plane()

        # Add custom event handlers for scrolling and keys
        self.AddObserver("KeyPressEvent", self.on_key_press)
        self.AddObserver("MouseWheelForwardEvent", self.on_mouse_wheel_forward)
        self.AddObserver("MouseWheelBackwardEvent",
                         self.on_mouse_wheel_backward)

    def reset_plane(self):
        """Create a new random plane and compute scalar field"""
        # Create random plane through a random point
        random_normal = np.random.randn(3).astype(np.float32)
        random_normal /= np.linalg.norm(random_normal)
        random_point = self.mesh.points[np.random.randint(
            0, len(self.mesh.points))]

        plane = tf.Plane.from_point_normal(random_point, random_normal)

        # Compute scalar field (signed distance to plane)
        self.scalars = tf.distance_field(self.mesh.points, plane)
        self.distance = 0.0

        # Cache min/max for adaptive level spacing
        self.min_d = float(np.min(self.scalars))
        self.max_d = float(np.max(self.scalars))
        self.spacing = (self.max_d - self.min_d) / 10.0  # N = 10 levels

    def compute_curves(self):
        """Extract isocontours and update visualization"""
        import time as time_module

        # Compute adaptive threshold levels
        N = 10  # Number of levels

        # Use fmod to wrap offset within spacing (infinite scrolling)
        wrapped_offset = self.distance % self.spacing
        if wrapped_offset < 0:
            wrapped_offset += self.spacing

        # Generate evenly spaced thresholds offset by wrapped_offset
        # Extend below and above the range so wrapping is seamless
        thresholds = np.array(
            [self.min_d + wrapped_offset + i *
                self.spacing for i in range(-1, N + 2)],
            dtype=self.mesh.dtype
        )

        # Extract isocontours with timing
        start_time = time_module.perf_counter()
        paths, curve_points = tf.isocontours(
            self.mesh, self.scalars, thresholds)
        elapsed = time_module.perf_counter() - start_time

        # Update timing
        self.times.add(elapsed)
        avg_time = self.times.get_average()
        self.text_actor.SetInput(
            f"Isocontours time: {format_time_ms(avg_time)}")

        # Convert curves to polydata (no tube filter needed!)
        if len(paths) > 0 and len(curve_points) > 0:
            poly = curves_to_polydata(paths, curve_points)
            self.curve_polydata.ShallowCopy(poly)
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
            self.compute_curves()
        else:
            # Pass to base class for camera controls
            vtk.vtkInteractorStyleTrackballCamera.OnKeyPress(self)

    def on_mouse_wheel_forward(self, obj, event):
        """Handle mouse wheel forward"""
        if self.GetInteractor().GetControlKey():
            # Move isocontours forward
            self.distance += self.spacing * 0.1
            self.compute_curves()
        else:
            # Pass to base class for zoom
            vtk.vtkInteractorStyleTrackballCamera.OnMouseWheelForward(self)

    def on_mouse_wheel_backward(self, obj, event):
        """Handle mouse wheel backward"""
        if self.GetInteractor().GetControlKey():
            # Move isocontours backward
            self.distance -= self.spacing * 0.1
            self.compute_curves()
        else:
            # Pass to base class for zoom
            vtk.vtkInteractorStyleTrackballCamera.OnMouseWheelBackward(self)


def main():
    # Parse command line arguments
    parser = create_parser("Interactive isocontour extraction", mesh_args=1)
    parser.epilog = """
Controls:
  N              Randomize cutting plane
  Ctrl + Scroll  Move isocontour levels
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
    transform = compute_centering_and_scaling_transform(
        points, target_radius=10.0)

    # Apply transformation
    points_homogeneous = np.hstack(
        [points, np.ones((len(points), 1), dtype=points.dtype)])
    points_transformed = (transform @ points_homogeneous.T).T[:, :3]

    # Create trueform mesh FIRST (this is the primary data structure)
    mesh = tf.Mesh(faces, points_transformed)
    mesh.build_tree()  # Build spatial tree for ray casting

    # Create VTK polydata for visualization (just a view)
    poly = numpy_to_polydata(points_transformed, faces)

    # Create renderers (main + text strip)
    renderer, renderer_text = create_renderer_with_text_strip()

    # Setup mesh actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.8, 0.8, 0.8)
    renderer.AddActor(actor)

    # Create VTK matrix for transformations
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    actor.SetUserMatrix(matrix)

    # Wrap in MeshData for interaction
    mesh_data = MeshData(mesh, actor, matrix)

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
        "Isocontours time: 0 ms",
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
    style = IsocontourInteractor(mesh_data, curve_poly, text_time)
    interactor.SetInteractorStyle(style)

    # Compute initial curves (now that interactor is attached)
    style.compute_curves()

    # Start
    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
