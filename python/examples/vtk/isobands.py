"""
Interactive isoband extraction example with VTK

Single viewport showing:
- Original mesh (semi-transparent)
- Isoband curves (teal)
- Filled isoband polygons (muted teal)

The scalar field is computed as signed distance from a plane.
Scroll to move the isoband levels. Press N to randomize the plane.
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

# Set to True to test dynamic mesh (OffsetBlockedArray)
USE_DYNAMIC_MESH = True


class IsobandInteractor(BaseInteractor):
    def __init__(self, mesh_data, isobands_polydata, curve_polydata, text_actor):
        super().__init__()

        self.mesh_data = mesh_data
        self.mesh = mesh_data.mesh
        self.isobands_polydata = isobands_polydata
        self.curve_polydata = curve_polydata
        self.text_actor = text_actor

        self.scalars = None
        self.distance = 0.0
        self.min_d = 0.0
        self.max_d = 0.0
        self.spacing = 0.1

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

        self.scalars = tf.distance(tf.Point(self.mesh.points), plane)
        self.distance = 0.0

        self.min_d = float(np.min(self.scalars))
        self.max_d = float(np.max(self.scalars))
        self.spacing = (self.max_d - self.min_d) / 10.0

    def randomize_plane(self):
        """Create random plane through a random point"""
        random_normal = np.random.randn(3).astype(self.mesh.dtype)
        random_normal /= np.linalg.norm(random_normal)
        random_point = self.mesh.points[np.random.randint(0, len(self.mesh.points))]

        plane = tf.Plane.from_point_normal(random_point, random_normal)

        self.scalars = tf.distance(tf.Point(self.mesh.points), plane)
        self.distance = 0.0

        self.min_d = float(np.min(self.scalars))
        self.max_d = float(np.max(self.scalars))
        self.spacing = (self.max_d - self.min_d) / 10.0

    def compute_bands(self):
        import time as time_module

        N = 10

        wrapped_offset = self.distance % self.spacing
        if wrapped_offset < 0:
            wrapped_offset += self.spacing

        cutvalues = np.array(
            [self.min_d + wrapped_offset + i * self.spacing for i in range(-1, N + 2)],
            dtype=self.mesh.dtype
        )

        parity = int(np.floor(self.distance / self.spacing)) & 1
        selected_bands = np.array([i for i in range(N + 2) if (i & 1) == parity], dtype=np.int32)

        start_time = time_module.perf_counter()
        (band_faces, band_points), labels, face_labels, (paths, curve_points) = tf.isobands(
            self.mesh, self.scalars, cutvalues, selected_bands, return_curves=True
        )
        elapsed = time_module.perf_counter() - start_time

        self.times.add(elapsed)
        avg_time = self.times.get_average()
        self.text_actor.SetInput(f"Isobands time: {format_time_ms(avg_time)}")

        if len(band_faces) > 0 and len(band_points) > 0:
            poly = numpy_to_polydata(band_points, band_faces)
            self.isobands_polydata.ShallowCopy(poly)
        else:
            self.isobands_polydata.Initialize()

        self.isobands_polydata.Modified()

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
            self.compute_bands()
        else:
            vtk.vtkInteractorStyleTrackballCamera.OnKeyPress(self)

    def on_mouse_wheel_forward(self, obj, event):
        if self.GetInteractor().GetControlKey():
            self.distance += self.spacing * 0.1
            self.compute_bands()
        else:
            vtk.vtkInteractorStyleTrackballCamera.OnMouseWheelForward(self)

    def on_mouse_wheel_backward(self, obj, event):
        if self.GetInteractor().GetControlKey():
            self.distance -= self.spacing * 0.1
            self.compute_bands()
        else:
            vtk.vtkInteractorStyleTrackballCamera.OnMouseWheelBackward(self)


def main():
    parser = create_parser("Interactive isoband extraction", mesh_args=1)
    parser.epilog = """
Controls:
  N              Randomize cutting plane
  Ctrl + Scroll  Move isoband levels
  Mouse drag     Rotate camera
"""
    args = parser.parse_args()
    mesh_file = args.mesh

    faces, points = tf.read_stl(mesh_file)

    if USE_DYNAMIC_MESH:
        faces = tf.as_offset_blocked(faces)

    transform = compute_centering_and_scaling_transform(points, target_radius=10.0)
    points_homogeneous = np.hstack([points, np.ones((len(points), 1), dtype=points.dtype)])
    points_transformed = (transform @ points_homogeneous.T).T[:, :3]

    mesh = tf.Mesh(faces, points_transformed)

    poly = numpy_to_polydata(points_transformed, faces)

    renderer, renderer_text = create_renderer_with_text_strip()

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

    isobands_poly = vtk.vtkPolyData()
    isobands_poly.Initialize()
    isobands_mapper = vtk.vtkPolyDataMapper()
    isobands_mapper.SetInputData(isobands_poly)
    isobands_actor = vtk.vtkActor()
    isobands_actor.SetMapper(isobands_mapper)
    isobands_actor.GetProperty().SetColor(0.0, 0.659, 0.604)
    renderer.AddActor(isobands_actor)

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
        "Isobands time: 0 ms",
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

    style = IsobandInteractor(mesh_data, isobands_poly, curve_poly, text_time)
    interactor.SetInteractorStyle(style)

    style.compute_bands()

    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
