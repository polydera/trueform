"""
Interactive point cloud alignment example with VTK

Demonstrates ICP alignment workflow:
- Drag source mesh (left click + drag)
- Rotate source mesh (right click + drag)
- Press 'A' to align (OBB coarse + ICP refinement)

Loads two dragon meshes at different resolutions and applies
Taubin smoothing to the target for variation.
"""

import vtk
import numpy as np
import trueform as tf
import time

from util import (
    BaseInteractor,
    MeshData,
    numpy_to_polydata,
    compute_centering_and_scaling_transform,
    random_rotation_matrix,
    create_text_actor,
    format_time_ms,
    get_camera_ray,
    NORMAL_COLOR,
    HIGHLIGHT_COLOR,
)


# Colors (teal-based scheme, matching C++ example)
SOURCE_COLOR = (0.0, 0.659, 0.604)      # Teal for source
SOURCE_HIGHLIGHT = (0.0, 0.75, 0.68)    # Slightly brighter teal on hover
TARGET_COLOR = (0.2, 0.35, 0.33)        # Dim teal for target
ALIGNED_COLOR = (0.0, 0.835, 0.745)     # Bright teal after alignment


def set_matrix_from_numpy(vtk_matrix, numpy_matrix):
    """Copy numpy 4x4 matrix to VTK matrix"""
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, float(numpy_matrix[i, j]))
    vtk_matrix.Modified()


def transform_points(pts, T):
    """Apply 4x4 transform to points"""
    ones = np.ones((len(pts), 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])
    return (T @ pts_h.T).T[:, :3].astype(np.float32)


class AlignmentInteractor(BaseInteractor):
    """Interactive alignment with drag, rotate, and alignment key"""

    def __init__(self, source_mesh_data, target_mesh_data, source_cloud, target_cloud, target_normals, text_actor, chamfer_text):
        super().__init__()

        self.source_mesh_data = source_mesh_data
        self.target_mesh_data = target_mesh_data
        self.source_cloud = source_cloud  # PointCloud for alignment
        self.target_cloud = target_cloud  # PointCloud for alignment
        self.target_normals = target_normals  # For point-to-plane ICP
        self.text_actor = text_actor
        self.chamfer_text = chamfer_text

        # Rotation state
        self.rotating = False
        self.last_pos = None
        self.rotation_center = None

        # Enable mesh interaction for source only (target is fixed)
        self.enable_mesh_interaction(
            [source_mesh_data],
            normal_color=SOURCE_COLOR,
            highlight_color=SOURCE_HIGHLIGHT
        )

        # Set target color
        target_mesh_data.actor.GetProperty().SetColor(*TARGET_COLOR)

        # Add key press and right mouse handlers
        self.AddObserver("KeyPressEvent", self.on_key_press)
        self.AddObserver("RightButtonPressEvent", self.on_right_down)
        self.AddObserver("RightButtonReleaseEvent", self.on_right_up)
        self.AddObserver("MouseMoveEvent", self.on_mouse_move)

        # Initial chamfer
        self.update_chamfer()

    def update_chamfer(self):
        """Update chamfer error display (subsampled for speed)"""
        source_pts = self.source_mesh_data.mesh.points[::10]
        T = self.source_cloud.transformation
        error = tf.chamfer_error((source_pts, T), self.target_cloud)
        self.chamfer_text.SetInput(f"Chamfer error: {error:.4f}")

    def on_mesh_dragged(self, mesh_data, delta):
        """Called when source mesh is dragged"""
        # Sync point cloud transformation with mesh
        self.source_cloud.transformation = self.source_mesh_data.mesh.transformation
        self.update_chamfer()

    def on_right_down(self, obj, event):
        """Start rotation on right mouse down - always rotates source mesh"""
        x, y = self.GetInteractor().GetEventPosition()

        self.rotating = True
        self.last_pos = (x, y)

        # Rotation center = mesh center in world space
        transform = self.source_mesh_data.mesh.transformation
        mesh_center = np.mean(self.source_mesh_data.mesh.points, axis=0)
        self.rotation_center = (transform @ np.append(mesh_center, 1.0))[:3]

        self.GetInteractor().GetRenderWindow().HideCursor()

    def on_right_up(self, obj, event):
        """End rotation"""
        if self.rotating:
            self.rotating = False
            self.GetInteractor().GetRenderWindow().ShowCursor()
            # Sync point cloud transformation with mesh
            self.source_cloud.transformation = self.source_mesh_data.mesh.transformation
            self.update_chamfer()
        else:
            vtk.vtkInteractorStyleTrackballCamera.OnRightButtonUp(self)

    def on_mouse_move(self, obj, event):
        """Handle mouse move for rotation"""
        if self.rotating:
            x, y = self.GetInteractor().GetEventPosition()
            dx = x - self.last_pos[0]
            dy = y - self.last_pos[1]
            self.last_pos = (x, y)

            # Convert mouse movement to rotation angles
            angle_x = dy * 0.5  # degrees
            angle_y = dx * 0.5

            # Get current transform
            current_transform = self.source_mesh_data.mesh.transformation.copy()

            # Create rotation matrices (around world X and Y axes)
            rx = np.radians(angle_x)
            ry = np.radians(angle_y)

            Rx = np.array([
                [1, 0, 0, 0],
                [0, np.cos(rx), -np.sin(rx), 0],
                [0, np.sin(rx), np.cos(rx), 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)

            Ry = np.array([
                [np.cos(ry), 0, np.sin(ry), 0],
                [0, 1, 0, 0],
                [-np.sin(ry), 0, np.cos(ry), 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)

            # Rotate around mesh center
            center = self.rotation_center
            T_to_origin = np.eye(4, dtype=np.float32)
            T_to_origin[:3, 3] = -center

            T_back = np.eye(4, dtype=np.float32)
            T_back[:3, 3] = center

            # Apply: translate to origin, rotate, translate back
            new_transform = T_back @ Ry @ Rx @ T_to_origin @ current_transform

            # Update transforms
            set_matrix_from_numpy(self.source_mesh_data.matrix, new_transform)
            self.source_mesh_data.mesh.transformation = new_transform
            self.source_cloud.transformation = new_transform
            self.update_chamfer()

            self.GetInteractor().Render()

    def on_key_press(self, obj, event):
        """Handle key press - 'A' for alignment"""
        key = self.GetInteractor().GetKeySym().lower()

        if key == 'a':
            self.run_alignment()
        elif key == 'r':
            self.randomize_source()
        else:
            vtk.vtkInteractorStyleTrackballCamera.OnKeyPress(self)

    def randomize_source(self):
        """Randomize source mesh position and orientation"""
        current_transform = self.source_mesh_data.mesh.transformation.copy()

        # Get mesh center
        mesh_center = np.mean(self.source_mesh_data.mesh.points, axis=0)
        center_world = (current_transform @ np.append(mesh_center, 1.0))[:3]

        # Random rotation
        R = random_rotation_matrix(dtype=np.float32)
        R_4x4 = np.eye(4, dtype=np.float32)
        R_4x4[:3, :3] = R

        # Random translation offset
        offset = (np.random.rand(3) - 0.5) * 10

        # Rotate around center, then translate
        T_to_origin = np.eye(4, dtype=np.float32)
        T_to_origin[:3, 3] = -center_world

        T_back = np.eye(4, dtype=np.float32)
        T_back[:3, 3] = center_world + offset

        new_transform = T_back @ R_4x4 @ T_to_origin @ current_transform

        set_matrix_from_numpy(self.source_mesh_data.matrix, new_transform)
        self.source_mesh_data.mesh.transformation = new_transform
        self.source_cloud.transformation = new_transform

        # Reset color
        self.source_mesh_data.actor.GetProperty().SetColor(*SOURCE_COLOR)

        self.update_chamfer()
        self.GetInteractor().Render()

    def run_alignment(self):
        """Run OBB + ICP alignment"""
        self.text_actor.SetInput("Aligning...")
        self.GetInteractor().Render()

        source_cloud = self.source_cloud
        target_cloud = self.target_cloud

        # Get current source transform
        T_source = source_cloud.transformation.copy()

        start = time.perf_counter()

        # Step 1: OBB coarse alignment
        T_obb_delta = tf.fit_obb_alignment(source_cloud, target_cloud)
        T_obb = T_obb_delta @ T_source
        source_cloud.transformation = T_obb

        # Step 2: ICP refinement (point-to-plane with target normals)
        T_icp_delta = tf.fit_icp_alignment(
            source_cloud,
            (target_cloud, self.target_normals),
            max_iterations=50,
            n_samples=1000,
            k=1,
        )
        T_final = T_icp_delta @ T_obb

        align_time = time.perf_counter() - start

        # Update source transformation
        source_cloud.transformation = T_final
        self.source_mesh_data.mesh.transformation = T_final
        set_matrix_from_numpy(self.source_mesh_data.matrix, T_final)

        # Update display
        self.source_mesh_data.actor.GetProperty().SetColor(*ALIGNED_COLOR)
        self.text_actor.SetInput(f"Aligned in {format_time_ms(align_time)}")
        self.update_chamfer()
        self.GetInteractor().Render()


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Interactive alignment demo")
    parser.add_argument("--source", help="Source STL mesh file (default: dragon-500k.stl)")
    parser.add_argument("--target", help="Target STL mesh file (default: dragon-50k.stl)")
    args = parser.parse_args()

    # Default path relative to this file: python/examples/vtk -> benchmarks/data
    benchmarks_data = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "benchmarks", "data"))

    source_path = args.source or os.path.join(benchmarks_data, "dragon-500k.stl")
    target_path = args.target or os.path.join(benchmarks_data, "dragon-50k.stl")

    # Load meshes
    print(f"Loading source mesh: {source_path}")
    source_faces, source_points = tf.read_stl(source_path)
    print(f"  {len(source_faces)} faces, {len(source_points)} points")

    print(f"Loading target mesh: {target_path}")
    target_faces, target_points = tf.read_stl(target_path)
    print(f"  {len(target_faces)} faces, {len(target_points)} points")

    # Apply Taubin smoothing to target (preserves volume)
    print("Smoothing target mesh (50 Taubin iterations)...")
    target_mesh_for_smooth = tf.Mesh(target_faces, target_points)
    target_points = tf.taubin_smoothed(target_mesh_for_smooth, iterations=50, lambda_=0.9)

    # Compute normals for point-to-plane ICP
    target_mesh_for_normals = tf.Mesh(target_faces, target_points)
    target_normals = tf.point_normals(target_mesh_for_normals)
    print("  Done.")

    # Compute centering/scaling for both meshes
    source_transform = compute_centering_and_scaling_transform(source_points, target_radius=10.0)
    target_transform = compute_centering_and_scaling_transform(target_points, target_radius=10.0)

    # Apply random initial misalignment to source
    R = random_rotation_matrix(dtype=np.float32)
    R_4x4 = np.eye(4, dtype=np.float32)
    R_4x4[:3, :3] = R
    offset = np.array([5.0, 3.0, 2.0], dtype=np.float32)
    T_offset = np.eye(4, dtype=np.float32)
    T_offset[:3, 3] = offset

    source_combined = T_offset @ R_4x4 @ source_transform

    # Create trueform meshes (for ray casting / visualization)
    # Both use local points + transformation
    source_mesh = tf.Mesh(source_faces, source_points)
    source_mesh.transformation = source_combined
    source_mesh.build_tree()

    target_mesh = tf.Mesh(target_faces, target_points)
    target_mesh.transformation = target_transform
    target_mesh.build_tree()

    # Create point clouds (for alignment / chamfer error)
    # Both use local points + transformation
    source_cloud = tf.PointCloud(source_points)
    source_cloud.transformation = source_combined
    source_cloud.build_tree()

    target_cloud = tf.PointCloud(target_points)
    target_cloud.transformation = target_transform
    target_cloud.build_tree()

    # Create VTK actors
    # Source: local points + transform matrix
    source_poly = numpy_to_polydata(source_points, source_faces)
    source_mapper = vtk.vtkPolyDataMapper()
    source_mapper.SetInputData(source_poly)
    source_actor = vtk.vtkActor()
    source_actor.SetMapper(source_mapper)
    source_actor.GetProperty().SetColor(*SOURCE_COLOR)

    source_matrix = vtk.vtkMatrix4x4()
    set_matrix_from_numpy(source_matrix, source_combined)
    source_actor.SetUserMatrix(source_matrix)

    # Target: local points + transform matrix
    target_poly = numpy_to_polydata(target_points, target_faces)
    target_mapper = vtk.vtkPolyDataMapper()
    target_mapper.SetInputData(target_poly)
    target_actor = vtk.vtkActor()
    target_actor.SetMapper(target_mapper)
    target_actor.GetProperty().SetColor(*TARGET_COLOR)
    target_actor.GetProperty().SetOpacity(0.4)

    target_matrix = vtk.vtkMatrix4x4()
    set_matrix_from_numpy(target_matrix, target_transform)
    target_actor.SetUserMatrix(target_matrix)

    # Create MeshData containers
    source_data = MeshData(source_mesh, source_actor, source_matrix)
    target_data = MeshData(target_mesh, target_actor, target_matrix)

    # Setup renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.12, 0.15)
    renderer.AddActor(source_actor)
    renderer.AddActor(target_actor)

    # Text strip at bottom
    renderer_text = vtk.vtkRenderer()
    renderer_text.SetViewport(0.0, 0.0, 1.0, 0.1)
    renderer_text.SetBackground(0.08, 0.1, 0.12)
    renderer_text.InteractiveOff()

    renderer.SetViewport(0.0, 0.1, 1.0, 1.0)

    # Text actors
    instructions = create_text_actor(
        "Drag: move | Right-drag: rotate | A: align | R: randomize",
        font_size=24,
        position=(0.02, 0.5),
        justification='left'
    )
    renderer_text.AddViewProp(instructions)

    chamfer_text = create_text_actor(
        "Chamfer error: --",
        font_size=24,
        position=(0.98, 0.5),
        justification='right'
    )
    renderer_text.AddViewProp(chamfer_text)

    # Render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.AddRenderer(renderer_text)
    render_window.SetSize(1200, 800)
    render_window.SetWindowName("Interactive Alignment - trueform")

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Custom interactor
    style = AlignmentInteractor(source_data, target_data, source_cloud, target_cloud, target_normals, instructions, chamfer_text)
    interactor.SetInteractorStyle(style)

    # Reset camera
    renderer.ResetCamera()

    # Start
    render_window.Render()
    print("\nControls:")
    print("  Left-drag on source mesh: translate")
    print("  Right-drag on source mesh: rotate")
    print("  A: run alignment (OBB + ICP)")
    print("  R: randomize source position")
    print()
    interactor.Start()


if __name__ == "__main__":
    main()
