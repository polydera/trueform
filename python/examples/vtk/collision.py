"""
Interactive collision detection example with VTK

Meshes are arranged in a 5x5 grid (25 instances total).
If fewer than 25 meshes are provided, they are cycled to fill the grid.
Times are averaged over the last 1000 frames.
"""

import vtk
import trueform as tf
import time

# Import utilities
from util import (
    load_mesh,
    load_mesh_shared,
    BaseInteractor,
    NORMAL_COLOR,
    HIGHLIGHT_COLOR,
    RollingAverage,
    format_time_us,
    create_text_actor,
    create_renderer_with_text_strip,
    create_parser,
)

# Collision-specific color (local to this example)
COLLIDING_COLOR = (0.8, 1.0, 1.0)  # Cyan


class MouseRaycastInteractor(BaseInteractor):
    """Interactive style with ray casting on mouse move and mesh dragging"""

    def __init__(self, mesh_data_list, collide_text=None):
        super().__init__()
        self.mesh_data_list = mesh_data_list

        # Collision tracking
        self.colliding_mesh_set = set()  # Set of MeshData currently colliding

        # Timing tracking
        self.collide_times = RollingAverage(maxlen=1000)
        self.collide_text = collide_text

        # Enable mesh interaction (handles all picking/dragging automatically!)
        self.enable_mesh_interaction(mesh_data_list)

    def update_timing_text(self):
        """Update timing text actors with average times"""
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

    # Override hooks from BaseInteractor
    def on_mesh_selected(self, mesh_data):
        """Called when mesh is selected - start collision detection"""
        pass  # Nothing extra needed on selection

    def on_mesh_dragged(self, mesh_data, delta):
        """Called when mesh is dragged - check collisions"""
        self.check_collisions()

    def on_mesh_released(self, mesh_data):
        """Called when mesh is released - clear collision highlighting"""
        self.colliding_mesh_set.clear()
        # Colors already reset by base interactor


def main():
    # Parse command line arguments
    parser = create_parser("Interactive collision detection (5x5 grid)", mesh_args="many")
    parser.epilog = """
Controls:
  Hover        Highlight mesh
  Mouse drag   Move mesh (colliding meshes turn cyan)
"""
    args = parser.parse_args()
    mesh_files = args.meshes

    # Load each unique mesh file once and build structures
    # Use dynamic=True to test OffsetBlockedArray
    source_meshes = {}
    for filename in mesh_files:
        mesh_data = load_mesh(filename, (0.0, 0.0, 0.0), target_radius=10.0, random_rotation=False, dynamic=True)
        mesh_data.mesh.build_tree()
        source_meshes[filename] = mesh_data

    # Create 5×5 grid of meshes using shared_view
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
            source = source_meshes[filename]

            # Use shared_view to share geometry and tree
            mesh = load_mesh_shared(source, (x, y, z), random_rotation=True)
            mesh.actor.GetProperty().SetColor(*NORMAL_COLOR)
            mesh_data_list.append(mesh)

            mesh_index += 1

    # Create renderers (main + text strip)
    renderer, renderer_text = create_renderer_with_text_strip()

    # Add mesh actors to main renderer
    for mesh_data in mesh_data_list:
        renderer.AddActor(mesh_data.actor)

    # Create text actor for timing
    collide_text = create_text_actor(
        "Collision time: 0.0 μs",
        font_size=38,
        position=(0.03, 0.50),
        justification='left'
    )
    renderer_text.AddViewProp(collide_text)

    # Setup render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.AddRenderer(renderer_text)
    render_window.SetSize(800, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Enable mouse ray casting and dragging
    style = MouseRaycastInteractor(mesh_data_list, collide_text)
    interactor.SetInteractorStyle(style)

    # Start
    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
