"""
Interactive cross-section example with VTK

Single viewport showing:
- Original mesh (semi-transparent)
- Cross-section curves (teal)
- Filled cross-section polygons (muted teal)

The scalar field is computed as signed distance from a plane.
Scroll to move the cutting plane. Press N to randomize the plane.
"""

import vtk
import numpy as np
import trueform as tf

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


class CrossSectionInteractor(BaseInteractor):
    """Interactor for cross-section visualization"""

    def __init__(self, mesh_data, fill_polydata, curve_polydata, text_actor):
        super().__init__()

        self.mesh_data = mesh_data
        self.mesh = mesh_data.mesh
        self.fill_polydata = fill_polydata
        self.curve_polydata = curve_polydata
        self.text_actor = text_actor

        self.scalars = None
        self.cut_value = 0.0
        self.min_d = 0.0
        self.max_d = 1.0

        self.times = RollingAverage(maxlen=100)

        self.reset_plane()

        self.AddObserver("KeyPressEvent", self.on_key_press)
        self.AddObserver("MouseWheelForwardEvent", self.on_mouse_wheel_forward)
        self.AddObserver("MouseWheelBackwardEvent", self.on_mouse_wheel_backward)

    def reset_plane(self):
        """Create initial plane with normal (1, 2, 1) through centroid"""
        normal = np.array([1.0, 2.0, 1.0], dtype=self.mesh.dtype)
        normal /= np.linalg.norm(normal)
        center = self.mesh.points.mean(axis=0)

        plane = tf.Plane.from_point_normal(center, normal)

        self.scalars = tf.distance_field(self.mesh.points, plane)
        self.min_d = float(np.min(self.scalars))
        self.max_d = float(np.max(self.scalars))
        self.cut_value = (self.min_d + self.max_d) * 0.5

    def randomize_plane(self):
        """Create random plane through a random point"""
        random_normal = np.random.randn(3).astype(self.mesh.dtype)
        random_normal /= np.linalg.norm(random_normal)
        random_point = self.mesh.points[np.random.randint(0, len(self.mesh.points))]

        plane = tf.Plane.from_point_normal(random_point, random_normal)

        self.scalars = tf.distance_field(self.mesh.points, plane)
        self.min_d = float(np.min(self.scalars))
        self.max_d = float(np.max(self.scalars))
        self.cut_value = (self.min_d + self.max_d) * 0.5

    def compute_cross_section(self):
        """Extract cross-section curves and triangulate"""
        import time as time_module

        start_time = time_module.perf_counter()

        paths, curve_points = tf.isocontours(self.mesh, self.scalars, self.cut_value)

        if len(paths) > 0 and len(curve_points) > 0:
            tri_faces, tri_points = tf.triangulated((paths, curve_points))
        else:
            tri_faces, tri_points = np.zeros((0, 3), dtype=np.int32), np.zeros((0, 3), dtype=self.mesh.dtype)

        elapsed = time_module.perf_counter() - start_time

        self.times.add(elapsed)
        avg_time = self.times.get_average()
        self.text_actor.SetInput(f"Cross-section time: {format_time_ms(avg_time)}")

        if len(tri_faces) > 0 and len(tri_points) > 0:
            poly = numpy_to_polydata(tri_points, tri_faces)
            self.fill_polydata.ShallowCopy(poly)
        else:
            self.fill_polydata.Initialize()
        self.fill_polydata.Modified()

        if len(paths) > 0 and len(curve_points) > 0:
            curve_poly = curves_to_polydata(paths, curve_points)
            self.curve_polydata.ShallowCopy(curve_poly)
        else:
            self.curve_polydata.Initialize()
        self.curve_polydata.Modified()

        if self.GetInteractor() is not None:
            self.GetInteractor().Render()

    def on_key_press(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == "n":
            self.randomize_plane()
            self.compute_cross_section()
        else:
            vtk.vtkInteractorStyleTrackballCamera.OnKeyPress(self)

    def on_mouse_wheel_forward(self, obj, event):
        if self.GetInteractor().GetControlKey():
            range_d = self.max_d - self.min_d
            margin = range_d * 0.01
            self.cut_value = min(self.max_d - margin, self.cut_value + 0.005 * range_d)
            self.compute_cross_section()
        else:
            vtk.vtkInteractorStyleTrackballCamera.OnMouseWheelForward(self)

    def on_mouse_wheel_backward(self, obj, event):
        if self.GetInteractor().GetControlKey():
            range_d = self.max_d - self.min_d
            margin = range_d * 0.01
            self.cut_value = max(self.min_d + margin, self.cut_value - 0.005 * range_d)
            self.compute_cross_section()
        else:
            vtk.vtkInteractorStyleTrackballCamera.OnMouseWheelBackward(self)


def main():
    parser = create_parser("Interactive cross-section", mesh_args=1)
    parser.epilog = """
Controls:
  N              Randomize cutting plane
  Ctrl + Scroll  Move cutting plane
  Mouse drag     Rotate camera
"""
    args = parser.parse_args()
    mesh_file = args.mesh

    faces, points = tf.read_stl(mesh_file)

    transform = compute_centering_and_scaling_transform(points, target_radius=10.0)
    points_homogeneous = np.hstack([points, np.ones((len(points), 1), dtype=points.dtype)])
    points_transformed = (transform @ points_homogeneous.T).T[:, :3]

    mesh = tf.Mesh(faces, points_transformed)

    poly = numpy_to_polydata(points_transformed, faces)

    renderer, renderer_text = create_renderer_with_text_strip()

    # Original mesh (semi-transparent)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.5, 0.5, 0.55)
    actor.GetProperty().SetOpacity(0.25)
    renderer.AddActor(actor)

    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    actor.SetUserMatrix(matrix)

    mesh_data = MeshData(mesh, actor, matrix)

    # Cross-section fill (muted teal)
    fill_poly = vtk.vtkPolyData()
    fill_poly.Initialize()
    fill_mapper = vtk.vtkPolyDataMapper()
    fill_mapper.SetInputData(fill_poly)
    fill_actor = vtk.vtkActor()
    fill_actor.SetMapper(fill_mapper)
    fill_actor.GetProperty().SetColor(0.0, 0.659, 0.604)
    renderer.AddActor(fill_actor)

    # Cross-section curves (brighter teal)
    curve_poly = vtk.vtkPolyData()
    curve_poly.Initialize()
    curve_mapper = vtk.vtkPolyDataMapper()
    curve_mapper.SetInputData(curve_poly)
    curve_actor = vtk.vtkActor()
    curve_actor.SetMapper(curve_mapper)
    curve_actor.GetProperty().SetColor(0.0, 0.835, 0.745)
    curve_actor.GetProperty().SetRenderLinesAsTubes(True)
    curve_actor.GetProperty().SetLineWidth(8.0)
    renderer.AddActor(curve_actor)

    text_time = create_text_actor(
        "Cross-section time: 0 ms",
        font_size=38,
        position=(0.03, 0.50),
        justification='left'
    )
    renderer_text.AddViewProp(text_time)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.AddRenderer(renderer_text)
    render_window.SetSize(800, 600)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    style = CrossSectionInteractor(mesh_data, fill_poly, curve_poly, text_time)
    interactor.SetInteractorStyle(style)

    style.compute_cross_section()

    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
