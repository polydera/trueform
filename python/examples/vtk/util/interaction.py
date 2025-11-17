"""
Interaction utilities for VTK examples
"""

import vtk
import numpy as np
import trueform as tf


# Standard colors for mesh visualization
NORMAL_COLOR = (0.8, 0.8, 0.8)          # Light gray
HIGHLIGHT_COLOR = (0.85, 0.85, 0.9)     # Very subtle blue tint (was 1.0, 0.9, 1.0)


class BaseInteractor(vtk.vtkInteractorStyleTrackballCamera):
    """
    Base interactor with common state management for VTK examples

    Provides common functionality:
    - Selected mesh tracking
    - Camera vs selection mode
    - Drag mode for mesh movement
    - Moving plane creation for dragging
    - Optional mesh interaction (hovering, selection, dragging)

    Subclasses should override event handlers as needed.
    """

    def __init__(self):
        super().__init__()

        # State management
        self.selected_mesh_data = None  # Currently selected mesh
        self.selected_mode = False      # Dragging mode
        self.camera_mode = False        # Camera rotation mode

        # Movement tracking
        self.moving_plane = None
        self.last_point = None

        # Mesh interaction (set via enable_mesh_interaction)
        self._interactive_meshes = None
        self._mesh_normal_color = NORMAL_COLOR
        self._mesh_highlight_color = HIGHLIGHT_COLOR

    def make_moving_plane(self, origin, renderer):
        """
        Create a plane perpendicular to camera through the picked point

        Parameters
        ----------
        origin : np.ndarray
            Point on the plane (usually the picked point)
        renderer : vtkRenderer
            Renderer to get camera from
        """
        camera = renderer.GetActiveCamera()
        cam_pos = np.array(camera.GetPosition(), dtype=np.float32)
        focal_pt = np.array(camera.GetFocalPoint(), dtype=np.float32)

        # Normal points from camera to focal point
        normal = focal_pt - cam_pos
        normal = normal / np.linalg.norm(normal)

        self.moving_plane = tf.Plane.from_point_normal(origin, normal)

    def update_mesh_transformation(self, mesh_data):
        """
        Sync VTK matrix to trueform mesh transformation

        Parameters
        ----------
        mesh_data : MeshData
            Mesh data container
        """
        vtk_matrix = mesh_data.matrix

        # Convert to numpy (4x4)
        transform = np.eye(4, dtype=np.float32)
        for i in range(4):
            for j in range(4):
                transform[i, j] = vtk_matrix.GetElement(i, j)

        # Set on trueform mesh
        mesh_data.mesh.transformation = transform

    def move_mesh(self, mesh_data, dx):
        """
        Move mesh by delta vector

        Parameters
        ----------
        mesh_data : MeshData
            Mesh data container
        dx : np.ndarray
            Delta vector (3,)
        """
        matrix = mesh_data.matrix
        matrix.SetElement(0, 3, matrix.GetElement(0, 3) + dx[0])
        matrix.SetElement(1, 3, matrix.GetElement(1, 3) + dx[1])
        matrix.SetElement(2, 3, matrix.GetElement(2, 3) + dx[2])
        matrix.Modified()

        # Update trueform transformation
        self.update_mesh_transformation(mesh_data)

    def enable_mesh_interaction(self, mesh_data_list, normal_color=None, highlight_color=None):
        """
        Enable hover/selection/dragging for one or more meshes

        This sets up the common pattern of:
        - Hovering over mesh highlights it
        - Click to start dragging
        - Release to stop dragging
        - Automatically finds closest mesh when multiple meshes are provided

        Call this from your subclass __init__ to enable mesh interaction.
        Then override the hooks to respond to events.

        Parameters
        ----------
        mesh_data_list : MeshData or list of MeshData
            The mesh(es) to make interactive. Can be a single MeshData or a list.
        normal_color : tuple, optional
            RGB color for normal state (default: NORMAL_COLOR)
        highlight_color : tuple, optional
            RGB color for highlighted state (default: HIGHLIGHT_COLOR)

        Example
        -------
        >>> # Single mesh
        >>> class MyInteractor(BaseInteractor):
        ...     def __init__(self, mesh_data):
        ...         super().__init__()
        ...         self.enable_mesh_interaction([mesh_data])
        ...
        >>> # Multiple meshes
        >>> class MyInteractor(BaseInteractor):
        ...     def __init__(self, mesh_data_list):
        ...         super().__init__()
        ...         self.enable_mesh_interaction(mesh_data_list)
        ...         # Now all meshes respond to mouse!
        ...
        ...     def on_mesh_dragged(self, mesh_data, delta):
        ...         # Called when any mesh is dragged
        ...         print(f"Dragged mesh by {delta}")
        """
        # Accept single mesh or list
        if not isinstance(mesh_data_list, list):
            mesh_data_list = [mesh_data_list]

        self._interactive_meshes = mesh_data_list
        self._mesh_normal_color = normal_color or NORMAL_COLOR
        self._mesh_highlight_color = highlight_color or HIGHLIGHT_COLOR

        # Set initial colors
        for mesh_data in mesh_data_list:
            mesh_data.actor.GetProperty().SetColor(*self._mesh_normal_color)

        # Attach event observers
        self.AddObserver("MouseMoveEvent", self._handle_mesh_mouse_move)
        self.AddObserver("LeftButtonPressEvent", self._handle_mesh_left_down)
        self.AddObserver("LeftButtonReleaseEvent", self._handle_mesh_left_up)

    def _handle_mesh_mouse_move(self, obj, event):
        """Internal handler for mouse move with mesh interaction"""
        x, y = self.GetInteractor().GetEventPosition()
        renderer = self.GetInteractor().FindPokedRenderer(x, y)
        ray = get_camera_ray(renderer, x, y)

        if not self.selected_mode and not self.camera_mode:
            # Hovering - check which mesh we hit
            hit_mesh_data, hit_point = self._ray_hit_multiple(ray, self._interactive_meshes)

            if hit_mesh_data is not None:
                # Reset previously highlighted mesh if different
                if self.selected_mesh_data != hit_mesh_data:
                    self._reset_all_mesh_colors()

                # Highlight hit mesh
                hit_mesh_data.actor.GetProperty().SetColor(*self._mesh_highlight_color)
                self.selected_mesh_data = hit_mesh_data

                # Prepare for potential drag
                self.make_moving_plane(hit_point, renderer)
                self.last_point = hit_point

                # Call hook for subclasses
                self.on_mesh_hover(hit_mesh_data, hit_point)
            else:
                # Not hovering - reset all
                self._reset_all_mesh_colors()
                self.selected_mesh_data = None
                self.on_mesh_hover(None, None)

            self.GetInteractor().Render()

        elif self.selected_mode:
            # Dragging mesh
            if self.moving_plane is not None and self.selected_mesh_data is not None:
                hit_result = tf.ray_cast(ray, self.moving_plane)
                if hit_result is not None:
                    t = hit_result
                    next_point = ray.origin + t * ray.direction
                    dx = next_point - self.last_point
                    self.last_point = next_point
                    self.move_mesh(self.selected_mesh_data, dx)

                    # Call hook for subclasses
                    self.on_mesh_dragged(self.selected_mesh_data, dx)

                    self.GetInteractor().Render()

        elif self.camera_mode:
            # Pass through to camera controls
            vtk.vtkInteractorStyleTrackballCamera.OnMouseMove(self)

    def _handle_mesh_left_down(self, obj, event):
        """Internal handler for left button down with mesh interaction"""
        if self.selected_mesh_data is not None:
            # Start dragging
            self.selected_mode = True
            self.GetInteractor().GetRenderWindow().HideCursor()
            self.on_mesh_selected(self.selected_mesh_data)
        else:
            # Start camera rotation
            self.camera_mode = True
            vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonDown(self)

    def _handle_mesh_left_up(self, obj, event):
        """Internal handler for left button up with mesh interaction"""
        if self.selected_mode:
            self.selected_mode = False
            self.GetInteractor().GetRenderWindow().ShowCursor()

            # Reset to normal colors
            self._reset_all_mesh_colors()

            # Re-highlight if still hovering over a mesh
            x, y = self.GetInteractor().GetEventPosition()
            renderer = self.GetInteractor().FindPokedRenderer(x, y)
            ray = get_camera_ray(renderer, x, y)
            hit_mesh_data, _ = self._ray_hit_multiple(ray, self._interactive_meshes)
            if hit_mesh_data is not None:
                hit_mesh_data.actor.GetProperty().SetColor(*self._mesh_highlight_color)
                self.selected_mesh_data = hit_mesh_data
            else:
                self.selected_mesh_data = None

            # Call hook for subclasses
            self.on_mesh_released(self.selected_mesh_data if hit_mesh_data else None)

            self.GetInteractor().Render()
        elif self.camera_mode:
            self.camera_mode = False
            vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonUp(self)

    # Hooks for subclasses to override
    def on_mesh_hover(self, mesh_data, hit_point):
        """Called when mouse hovers over/off the mesh"""
        pass

    def on_mesh_selected(self, mesh_data):
        """Called when mesh is clicked (drag starts)"""
        pass

    def on_mesh_dragged(self, mesh_data, delta):
        """Called during mesh dragging"""
        pass

    def on_mesh_released(self, mesh_data):
        """Called when mesh is released (drag ends)"""
        pass

    def _ray_hit_multiple(self, ray, mesh_data_list):
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

    def _reset_all_mesh_colors(self):
        """Reset all interactive meshes to normal color"""
        if self._interactive_meshes:
            for mesh_data in self._interactive_meshes:
                mesh_data.actor.GetProperty().SetColor(*self._mesh_normal_color)


def get_camera_ray(renderer, x, y):
    """
    Get ray from camera through mouse position

    Parameters
    ----------
    renderer : vtkRenderer
        VTK renderer
    x, y : int
        Mouse position in display coordinates

    Returns
    -------
    tf.Ray
        Ray from camera through mouse position
    """
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
