"""
Interactive ray casting and collision detection example with VTK

Features:
- Mouse over meshes to highlight them using trueform ray casting
- Click and drag to move meshes
- Colliding meshes turn cyan during drag
- Meshes are randomly rotated on initialization
- Real-time performance metrics displayed on screen (ray picking and collision detection times)

Usage:
    python collision.py mesh1.stl [mesh2.stl mesh3.stl ...]

Meshes are arranged in a 5×5 grid (25 instances total).
If fewer than 25 meshes are provided, they are cycled to fill the grid.
Times are averaged over the last 1000 frames.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import vtk
import numpy as np
import trueform as tf
import time

# Import utilities
from util import (
    MeshData,
    load_mesh,
    BaseInteractor,
    get_camera_ray,
    NORMAL_COLOR,
    HIGHLIGHT_COLOR,
    RollingAverage,
    format_time_us,
    create_text_actor,
    create_renderer_with_text_strip,
)

# Collision-specific color (local to this example)
COLLIDING_COLOR = (0.8, 1.0, 1.0)  # Cyan


def ray_hit_multiple(ray, mesh_data_list):
    """
    Cast ray against all meshes, return closest hit

    Uses ray_config optimization: after each hit, update max_t to prune subsequent searches
    """
    closest_t = np.inf
    hit_mesh_data = None
    hit_point = None

    # Initialize config with default range
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


class MouseRaycastInteractor(BaseInteractor):
    """Interactive style with ray casting on mouse move and mesh dragging"""

    def __init__(self, mesh_data_list, pick_text=None, collide_text=None):
        super().__init__()
        self.mesh_data_list = mesh_data_list

        # Collision tracking
        self.colliding_mesh_set = set()  # Set of MeshData currently colliding

        # Timing tracking (rolling averages over last 1000 frames)
        self.pick_times = RollingAverage(maxlen=1000)
        self.collide_times = RollingAverage(maxlen=1000)
        self.pick_text = pick_text
        self.collide_text = collide_text

        self.AddObserver("MouseMoveEvent", self.on_mouse_move)
        self.AddObserver("LeftButtonPressEvent", self.on_left_button_down)
        self.AddObserver("LeftButtonReleaseEvent", self.on_left_button_up)

    def reset_all_colors(self):
        """Reset all meshes to normal color"""
        for mesh_data in self.mesh_data_list:
            mesh_data.actor.GetProperty().SetColor(*NORMAL_COLOR)

    def update_timing_text(self):
        """Update timing text actors with average times"""
        if self.pick_text and len(self.pick_times) > 0:
            avg_pick = self.pick_times.get_average()
            self.pick_text.SetInput(f"Ray picking time: {format_time_us(avg_pick)}")

        if self.collide_text and len(self.collide_times) > 0:
            avg_collide = self.collide_times.get_average()
            self.collide_text.SetInput(f"Collision time: {format_time_us(avg_collide)}")

    def check_collisions(self):
        """Check if selected mesh collides with others"""
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
        self.collide_times.add(elapsed)
        self.update_timing_text()

        # Update colors
        for mesh_data in self.mesh_data_list:
            if mesh_data == self.selected_mesh_data:
                # Keep selected mesh highlighted
                mesh_data.actor.GetProperty().SetColor(*HIGHLIGHT_COLOR)
            elif mesh_data in self.colliding_mesh_set:
                # Color colliding meshes cyan
                mesh_data.actor.GetProperty().SetColor(*COLLIDING_COLOR)
            else:
                # Reset non-colliding meshes
                mesh_data.actor.GetProperty().SetColor(*NORMAL_COLOR)

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
                hit_mesh_data.actor.GetProperty().SetColor(*HIGHLIGHT_COLOR)
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
            self.pick_times.add(elapsed)
            self.update_timing_text()

            if hit_mesh_data is not None:
                # Reset previously highlighted mesh
                if self.selected_mesh_data != hit_mesh_data:
                    self.reset_all_colors()

                # Highlight new mesh
                hit_mesh_data.actor.GetProperty().SetColor(*HIGHLIGHT_COLOR)
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


def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python collision.py mesh1.stl [mesh2.stl mesh3.stl ...]")
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
            mesh = load_mesh(filename, (x, y, z), target_radius=10.0, random_rotation=True)
            mesh.mesh.build_tree()
            mesh.actor.GetProperty().SetColor(*NORMAL_COLOR)
            mesh_data_list.append(mesh)

            mesh_index += 1

    # Create renderers (main + text strip)
    renderer, renderer_text = create_renderer_with_text_strip()

    # Add mesh actors to main renderer
    for mesh_data in mesh_data_list:
        renderer.AddActor(mesh_data.actor)

    # Create text actors for bottom strip
    pick_text = create_text_actor(
        "Ray picking time: 0.0 μs",
        font_size=40,
        position=(0.03, 0.30),
        justification='left'
    )
    renderer_text.AddViewProp(pick_text)

    collide_text = create_text_actor(
        "Collision time: 0.0 μs",
        font_size=40,
        position=(0.03, 0.70),
        justification='left'
    )
    renderer_text.AddViewProp(collide_text)

    text_help = create_text_actor(
        "Hover to highlight.\nClick and drag to move meshes.\nColliding meshes turn cyan.",
        font_size=40,
        position=(0.97, 0.50),
        justification='right'
    )
    text_help.GetTextProperty().SetLineSpacing(1.5)
    renderer_text.AddViewProp(text_help)

    # Setup render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.AddRenderer(renderer_text)
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
