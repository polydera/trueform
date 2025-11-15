"""
Interactive ray casting and collision detection example with VTK

Features:
- Mouse over meshes to highlight them using trueform ray casting
- Click and drag to move meshes
- Colliding meshes turn cyan during drag
- Meshes are randomly rotated on initialization
- Real-time performance metrics displayed on screen (ray picking and collision detection times)

Usage:
    python mouse_raycast.py mesh1.stl [mesh2.stl mesh3.stl ...]

Meshes are arranged in a 5×5 grid (25 instances total).
If fewer than 25 meshes are provided, they are cycled to fill the grid.
Times are averaged over the last 1000 frames.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import vtk
from vtk.util import numpy_support
import numpy as np
import trueform as tf
import time
from collections import deque
from scipy.spatial.transform import Rotation


class MeshData:
    """Container for mesh rendering data"""
    def __init__(self, mesh, actor, matrix):
        self.mesh = mesh        # tf.Mesh
        self.actor = actor      # vtkActor
        self.matrix = matrix    # vtkMatrix4x4


def numpy_to_polydata(points: np.ndarray, triangles: np.ndarray) -> vtk.vtkPolyData:
    """
    Convert numpy arrays to VTK PolyData (VTK 9+ format)

    points:    (m, 3) float array
    triangles: (n, 3) int array of vertex indices (0-based)
    """
    # Points
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(
        numpy_support.numpy_to_vtk(
            num_array=points,
            deep=False,
            array_type=vtk.VTK_FLOAT if points.dtype == np.float32 else vtk.VTK_DOUBLE,
        )
    )

    # CellArray (triangles via offsets + connectivity)
    n_tris = triangles.shape[0]

    # Flat connectivity: [i0, i1, i2, j0, j1, j2, ...]
    connectivity = triangles.astype(np.int64).ravel()

    # Offsets: start index of each cell in connectivity, plus final total
    offsets = np.arange(0, 3 * n_tris + 1, 3, dtype=np.int64)

    vtk_conn = numpy_support.numpy_to_vtkIdTypeArray(connectivity, deep=True)
    vtk_offs = numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True)

    cell_array = vtk.vtkCellArray()
    cell_array.SetData(vtk_offs, vtk_conn)

    # PolyData
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(cell_array)
    return poly


def compute_centering_and_scaling_transform(points: np.ndarray, target_radius: float = 10.0) -> np.ndarray:
    """
    Compute 4x4 transformation matrix to center and scale mesh

    Mimics collision.cpp center_and_scale() function:
    - Centers mesh at origin
    - Scales to fit in sphere of given radius
    """
    # Compute AABB
    min_pt = np.min(points, axis=0)
    max_pt = np.max(points, axis=0)
    center = (min_pt + max_pt) / 2.0

    # Compute bounding sphere radius (diagonal / 2)
    diagonal = max_pt - min_pt
    radius = np.linalg.norm(diagonal) / 2.0

    # Scale factor to fit in target radius
    scale = target_radius / radius

    # Create transform: first translate to origin, then scale
    T = np.eye(4, dtype=np.float32)
    T[0:3, 0:3] *= scale
    T[0:3, 3] = -center * scale

    return T


def get_camera_ray(renderer, x, y):
    """Get ray from camera through mouse position"""
    camera = renderer.GetActiveCamera()

    # Get depth at focal point
    renderer.SetWorldPoint(*camera.GetFocalPoint(), 1.0)
    renderer.WorldToDisplay()
    depth = renderer.GetDisplayPoint()[2]

    # Convert mouse to world coordinates
    renderer.SetDisplayPoint(x, y, depth)
    renderer.DisplayToWorld()
    world_pt = np.array(renderer.GetWorldPoint()[:3], dtype=np.float32)
    world_pt /= renderer.GetWorldPoint()[3]  # Homogeneous divide

    # Ray from camera through world point
    cam_pos = np.array(camera.GetPosition(), dtype=np.float32)
    direction = world_pt - cam_pos

    return tf.Ray(origin=cam_pos, direction=direction)


def ray_hit_multiple(ray, mesh_data_list):
    """
    Cast ray against all meshes, return closest hit

    Similar to geometry_handle_t::ray_hit() in collision.cpp
    Uses ray_config optimization: after each hit, update max_t to prune subsequent searches
    """
    closest_t = np.inf
    hit_mesh_data = None
    hit_point = None

    # Initialize config with default range (np.inf is supported)
    config = (0.0, np.inf)

    for mesh_data in mesh_data_list:
        result = tf.ray_cast(ray, mesh_data.mesh, config)
        if result is not None:
            face_idx, t = result
            if t < closest_t:
                closest_t = t
                hit_mesh_data = mesh_data
                hit_point = ray.origin + t * ray.direction
                # Update config to only check up to current closest hit
                config = (0.0, closest_t)

    return hit_mesh_data, hit_point


class MouseRaycastInteractor(vtk.vtkInteractorStyleTrackballCamera):
    """Interactive style with ray casting on mouse move and mesh dragging"""

    def __init__(self, mesh_data_list, pick_text=None, collide_text=None):
        self.mesh_data_list = mesh_data_list

        # State management
        self.selected_mesh_data = None  # Currently selected mesh
        self.selected_mode = False      # Dragging the mesh
        self.camera_mode = False        # Rotating camera

        # Movement tracking
        self.moving_plane = None
        self.last_point = None

        # Collision tracking
        self.colliding_mesh_set = set()  # Set of MeshData currently colliding

        # Timing tracking (rolling averages over last 1000 frames)
        self.pick_times = deque(maxlen=1000)
        self.collide_times = deque(maxlen=1000)
        self.pick_text = pick_text
        self.collide_text = collide_text

        # Colors
        self.normal_color = (0.8, 0.8, 0.8)
        self.highlight_color = (1.0, 0.9, 1.0)  # Pinkish-white
        self.colliding_color = (0.8, 1.0, 1.0)  # Cyan

        self.AddObserver("MouseMoveEvent", self.on_mouse_move)
        self.AddObserver("LeftButtonPressEvent", self.on_left_button_down)
        self.AddObserver("LeftButtonReleaseEvent", self.on_left_button_up)

    def make_moving_plane(self, origin, renderer):
        """Create a plane perpendicular to camera through the picked point"""
        camera = renderer.GetActiveCamera()
        cam_pos = np.array(camera.GetPosition(), dtype=np.float32)
        focal_pt = np.array(camera.GetFocalPoint(), dtype=np.float32)

        # Normal points from camera to focal point
        normal = focal_pt - cam_pos
        normal = normal / np.linalg.norm(normal)

        self.moving_plane = tf.Plane.from_point_normal(origin, normal)

    def update_mesh_transformation(self, mesh_data):
        """Sync VTK matrix to trueform mesh transformation"""
        vtk_matrix = mesh_data.matrix

        # Convert to numpy (4x4)
        transform = np.eye(4, dtype=np.float32)
        for i in range(4):
            for j in range(4):
                transform[i, j] = vtk_matrix.GetElement(i, j)

        # Set on trueform mesh
        mesh_data.mesh.transformation = transform

    def move_mesh(self, mesh_data, dx):
        """Move mesh by delta vector"""
        matrix = mesh_data.matrix
        matrix.SetElement(0, 3, matrix.GetElement(0, 3) + dx[0])
        matrix.SetElement(1, 3, matrix.GetElement(1, 3) + dx[1])
        matrix.SetElement(2, 3, matrix.GetElement(2, 3) + dx[2])
        matrix.Modified()

        # Update trueform transformation
        self.update_mesh_transformation(mesh_data)

    def reset_all_colors(self):
        """Reset all meshes to normal color"""
        for mesh_data in self.mesh_data_list:
            mesh_data.actor.GetProperty().SetColor(*self.normal_color)

    def update_timing_text(self):
        """Update timing text actors with average times"""
        if self.pick_text and len(self.pick_times) > 0:
            avg_pick = sum(self.pick_times) / len(self.pick_times)
            self.pick_text.SetInput(f"Ray picking time: {avg_pick * 1e6:.1f} μs")

        if self.collide_text and len(self.collide_times) > 0:
            avg_collide = sum(self.collide_times) / len(self.collide_times)
            self.collide_text.SetInput(f"Collision time: {avg_collide * 1e6:.1f} μs")

    def check_collisions(self):
        """
        Check if selected mesh collides with others

        Similar to geometry_handle_t::intersects_any() in collision.cpp
        """
        if self.selected_mesh_data is None:
            return

        # Clear previous collisions
        self.colliding_mesh_set.clear()

        # Time the collision detection
        start_time = time.perf_counter()

        # Check against all other meshes
        for other_mesh_data in self.mesh_data_list:
            if other_mesh_data == self.selected_mesh_data:
                continue

            # Check intersection
            collision = tf.intersects(self.selected_mesh_data.mesh, other_mesh_data.mesh)
            if collision:
                self.colliding_mesh_set.add(other_mesh_data)

        # Record timing
        elapsed = time.perf_counter() - start_time
        self.collide_times.append(elapsed)
        self.update_timing_text()

        # Update colors
        for mesh_data in self.mesh_data_list:
            if mesh_data == self.selected_mesh_data:
                # Keep selected mesh highlighted
                mesh_data.actor.GetProperty().SetColor(*self.highlight_color)
            elif mesh_data in self.colliding_mesh_set:
                # Color colliding meshes cyan
                mesh_data.actor.GetProperty().SetColor(*self.colliding_color)
            else:
                # Reset non-colliding meshes
                mesh_data.actor.GetProperty().SetColor(*self.normal_color)

    def on_left_button_down(self, obj, event):
        """Handle left button press"""
        if self.selected_mesh_data is not None:
            # Enter drag mode
            self.selected_mode = True
            self.GetInteractor().GetRenderWindow().HideCursor()
        else:
            # Enter camera mode
            self.camera_mode = True
            vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonDown(self)

    def on_left_button_up(self, obj, event):
        """Handle left button release"""
        if self.selected_mode:
            self.selected_mode = False
            self.GetInteractor().GetRenderWindow().ShowCursor()

            # Clear collision highlighting
            self.colliding_mesh_set.clear()
            self.reset_all_colors()

            # Re-highlight if still hovering over a mesh
            x, y = self.GetInteractor().GetEventPosition()
            renderer = self.GetInteractor().FindPokedRenderer(x, y)
            ray = get_camera_ray(renderer, x, y)
            hit_mesh_data, _ = ray_hit_multiple(ray, self.mesh_data_list)
            if hit_mesh_data is not None:
                hit_mesh_data.actor.GetProperty().SetColor(*self.highlight_color)
                self.selected_mesh_data = hit_mesh_data
            else:
                self.selected_mesh_data = None

            self.GetInteractor().Render()
        elif self.camera_mode:
            self.camera_mode = False
            vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonUp(self)

    def on_mouse_move(self, obj, event):
        """Handle mouse move"""
        x, y = self.GetInteractor().GetEventPosition()
        renderer = self.GetInteractor().FindPokedRenderer(x, y)
        ray = get_camera_ray(renderer, x, y)

        if not self.selected_mode and not self.camera_mode:
            # Ray casting and highlighting (with timing)
            start_time = time.perf_counter()
            hit_mesh_data, hit_point = ray_hit_multiple(ray, self.mesh_data_list)
            elapsed = time.perf_counter() - start_time
            self.pick_times.append(elapsed)
            self.update_timing_text()

            if hit_mesh_data is not None:
                # Reset previously highlighted mesh
                if self.selected_mesh_data != hit_mesh_data:
                    self.reset_all_colors()

                # Highlight new mesh
                hit_mesh_data.actor.GetProperty().SetColor(*self.highlight_color)
                self.selected_mesh_data = hit_mesh_data

                # Create moving plane for potential drag
                self.make_moving_plane(hit_point, renderer)
                self.last_point = hit_point
            else:
                # Reset all to normal
                self.reset_all_colors()
                self.selected_mesh_data = None

            self.GetInteractor().Render()

        elif self.selected_mode:
            # Dragging mesh
            if self.moving_plane is not None and self.selected_mesh_data is not None:
                # Project ray onto moving plane
                hit_result = tf.ray_cast(ray, self.moving_plane)
                if hit_result is not None:
                    t = hit_result
                    next_point = ray.origin + t * ray.direction
                    dx = next_point - self.last_point
                    self.last_point = next_point
                    self.move_mesh(self.selected_mesh_data, dx)

                    # Check for collisions and update colors
                    self.check_collisions()

                    self.GetInteractor().Render()

        elif self.camera_mode:
            # Camera rotation
            vtk.vtkInteractorStyleTrackballCamera.OnMouseMove(self)


def load_mesh(filename, position):
    """
    Load mesh from file, center, scale, position, and apply random rotation

    Args:
        filename: Path to mesh file
        position: (x, y, z) position in world coordinates

    Returns MeshData with configured mesh, actor, and matrix
    """
    # Read STL with trueform
    faces, points = tf.read_stl(filename)

    # Compute centering and scaling transform
    center_scale_transform = compute_centering_and_scaling_transform(points, target_radius=10.0)

    # Create random rotation transform
    random_rotation = Rotation.random().as_matrix().astype(np.float32)
    rotation_transform = np.eye(4, dtype=np.float32)
    rotation_transform[:3, :3] = random_rotation

    # Create position transform (translation)
    position_transform = np.eye(4, dtype=np.float32)
    position_transform[0, 3] = position[0]  # X offset
    position_transform[1, 3] = position[1]  # Y offset
    position_transform[2, 3] = position[2]  # Z offset

    # Combine: first center/scale, then rotate, then position
    combined_transform = position_transform @ rotation_transform @ center_scale_transform

    # Create trueform mesh with transformation
    mesh = tf.Mesh(faces, points)
    mesh.transformation = combined_transform

    # Create VTK visualization (uses original points)
    polydata = numpy_to_polydata(points, faces)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.8, 0.8, 0.8)

    # Set VTK transformation matrix
    matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix.SetElement(i, j, combined_transform[i, j])
    actor.SetUserMatrix(matrix)

    return MeshData(mesh, actor, matrix)


def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python mouse_raycast.py mesh1.stl [mesh2.stl mesh3.stl ...]")
        sys.exit(1)

    mesh_files = sys.argv[1:]

    # Create 5×5 grid of meshes
    grid_size = 5
    spacing = 15.0  # Distance between meshes
    mesh_data_list = []

    mesh_index = 0
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate grid position (centered around origin)
            x = i * spacing - (grid_size - 1) * spacing / 2.0
            y = j * spacing - (grid_size - 1) * spacing / 2.0
            z = 0.0

            # Cycle through available mesh files
            filename = mesh_files[mesh_index % len(mesh_files)]
            mesh = load_mesh(filename, (x, y, z))
            mesh.mesh.build_tree()
            mesh_data_list.append(mesh)

            mesh_index += 1

    # Create timing text actors
    pick_text = vtk.vtkTextActor()
    pick_text.SetInput("Ray picking time: 0.0 μs")
    pick_text_prop = pick_text.GetTextProperty()
    pick_text_prop.SetFontSize(24)
    pick_text_prop.SetColor(1.0, 1.0, 1.0)
    pick_text_prop.SetJustificationToLeft()
    pick_text.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    pick_text.SetPosition(0.02, 0.95)  # Top-left corner

    collide_text = vtk.vtkTextActor()
    collide_text.SetInput("Collision time: 0.0 μs")
    collide_text_prop = collide_text.GetTextProperty()
    collide_text_prop.SetFontSize(24)
    collide_text_prop.SetColor(1.0, 1.0, 1.0)
    collide_text_prop.SetJustificationToLeft()
    collide_text.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    collide_text.SetPosition(0.02, 0.90)  # Below pick time

    # Setup scene
    renderer = vtk.vtkRenderer()
    for mesh_data in mesh_data_list:
        renderer.AddActor(mesh_data.actor)
    renderer.AddViewProp(pick_text)
    renderer.AddViewProp(collide_text)
    renderer.SetBackground(27.0/255, 43.0/255, 52.0/255)  # Dark blue-grey

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Enable mouse ray casting and dragging
    style = MouseRaycastInteractor(mesh_data_list, pick_text, collide_text)
    interactor.SetInteractorStyle(style)

    # Start
    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
