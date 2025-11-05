#pragma once
#include "../../util/read_mesh.hpp"
#include "./data_bridge.hpp"
#include "trueform/trueform.hpp"
#include "vtkCamera.h"
#include "vtkCommand.h"
#include "vtkPolyData.h"
#include "vtkProperty.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkSTLReader.h"
#include "vtkTextActor.h"
#include <filesystem>
#include <memory>

template <typename T> struct vtk_deleter {
  void operator()(T *ptr) const {
    if (ptr) {
      ptr->Delete();
    }
  }
};

template <typename T> using vtk_unique_ptr = std::unique_ptr<T, vtk_deleter<T>>;

template <typename T, typename... Ts> auto vtk_make_unique(Ts &&...args) {
  return vtk_unique_ptr<T>{T::New(static_cast<Ts &&>(args)...)};
}

inline auto readSTL(std::string name) {
  auto reader = vtk_make_unique<vtkSTLReader>();
  reader->SetFileName(name.c_str());
  reader->Update();
  auto out = vtk_make_unique<vtkPolyData>();
  out->ShallowCopy(reader->GetOutput());
  return out;
}

inline auto to_polydata(const tf::polygons_buffer<int, float, 3, 3> &polys) {
  auto cells = vtk_make_unique<vtkCellArray>();
  cells->Initialize();
  auto offsets = cells->GetOffsetsArray();
  offsets->SetNumberOfComponents(1);
  offsets->SetNumberOfTuples(polys.faces().size() + 1);
  auto offsets_r =
      tf::make_range(static_cast<vtkIdType *>(offsets->GetVoidPointer(0)),
                     offsets->GetNumberOfValues());
  tf::parallel_apply(tf::enumerate(offsets_r), [](auto pair) {
    auto &&[id, offset] = pair;
    offset = 3 * id;
  });
  auto ids = cells->GetConnectivityArray();
  ids->SetNumberOfComponents(3);
  ids->SetNumberOfTuples(polys.faces().size());
  auto ids_r = tf::make_range(static_cast<vtkIdType *>(ids->GetVoidPointer(0)),
                              offsets->GetNumberOfValues());
  tf::parallel_copy(polys.faces_buffer().data_buffer(), ids_r);
  auto points = vtk_make_unique<vtkPoints>();
  points->SetNumberOfPoints(polys.points().size());
  tf::parallel_copy(polys.points(), get_points(points.get()));
  auto out = vtk_make_unique<vtkPolyData>();
  out->SetPoints(points.get());
  out->SetPolys(cells.get());
  return out;
}

inline auto readOBJ(std::string name) {
  auto [raw_points, raw_triangle_faces] = tf::examples::read_mesh(name);
  auto points_r = tf::make_points<3>(raw_points);
  auto faces_r = tf::make_blocked_range<3>(raw_triangle_faces);
  auto cells = vtk_make_unique<vtkCellArray>();
  cells->Initialize();
  auto offsets = cells->GetOffsetsArray();
  offsets->SetNumberOfComponents(1);
  offsets->SetNumberOfTuples(faces_r.size() + 1);
  auto offsets_r =
      tf::make_range(static_cast<vtkIdType *>(offsets->GetVoidPointer(0)),
                     offsets->GetNumberOfValues());
  tf::parallel_apply(tf::enumerate(offsets_r), [](auto pair) {
    auto &&[id, offset] = pair;
    offset = 3 * id;
  });
  auto ids = cells->GetConnectivityArray();
  ids->SetNumberOfComponents(3);
  ids->SetNumberOfTuples(faces_r.size());
  auto ids_r = tf::make_range(static_cast<vtkIdType *>(ids->GetVoidPointer(0)),
                              offsets->GetNumberOfValues());
  tf::parallel_copy(raw_triangle_faces, ids_r);
  auto points = vtk_make_unique<vtkPoints>();
  points->SetNumberOfPoints(points_r.size());
  tf::parallel_copy(points_r, get_points(points.get()));
  auto out = vtk_make_unique<vtkPolyData>();
  out->SetPoints(points.get());
  out->SetPolys(cells.get());
  return out;
}

inline auto read_mesh(std::string filename) {
  std::filesystem::path path{filename};

  if (!path.has_extension()) {
    return vtk_make_unique<vtkPolyData>();
  }
  if (path.extension() == ".stl") {
    return readSTL(filename);
  }
  if (path.extension() == ".obj") {
    return readOBJ(filename);
  }
  return vtk_make_unique<vtkPolyData>();
}

inline auto get_world_point_ray(vtkRenderer *renderer, int x, int y) {
  auto *camera = renderer->GetActiveCamera();
  tf::ray<double, 3> ray;
  if (!camera)
    return ray;

  auto origin = tf::make_vector_view<3>(camera->GetFocalPoint());
  auto ray_pos = tf::make_vector_view<3>(&ray.origin[0]);
  auto ray_dir = tf::make_vector_view<3>(&ray.direction[0]);
  if (camera->GetParallelProjection()) {
    renderer->SetDisplayPoint(x, y, 0.0);
    renderer->DisplayToWorld();
    ray_pos = tf::make_vector_view<3>(renderer->GetWorldPoint());
    ray_dir = tf::make_vector_view<3>(camera->GetDirectionOfProjection());
  } else {
    renderer->SetWorldPoint(origin[0], origin[1], origin[2], 1.0);
    renderer->WorldToDisplay();

    auto *display_coordinates = renderer->GetDisplayPoint();
    renderer->SetDisplayPoint(x, y, display_coordinates[2]);
    renderer->DisplayToWorld();

    auto *world_point = renderer->GetWorldPoint();
    auto *camera_position = camera->GetPosition();
    ray_pos = tf::make_vector_view<3>(world_point);
    ray_pos /= world_point[3];
    ray_dir = ray_pos - tf::make_vector_view<3>(camera_position);
    tf::normalize(ray_dir);
    ray_pos = tf::make_vector_view<3>(camera_position);
  }
  return ray;
}

class RightAlignTextUpdater : public vtkCommand {
public:
  vtkRenderWindow *Window;
  vtkTextActor *Text;
  int x;
  int y;

  RightAlignTextUpdater(vtkRenderWindow *Window, vtkTextActor *Text, int x,
                        int y)
      : Window{Window}, Text{Text}, x{x}, y{y} {}

  static RightAlignTextUpdater *New(vtkRenderWindow *Window, vtkTextActor *Text,
                                    int x, int y) {
    return new RightAlignTextUpdater{Window, Text, x, y};
  }

  void Execute(vtkObject *, unsigned long, void *) override {
    int *size = Window->GetSize();
    Text->SetDisplayPosition(size[0] - x, y);
  }
};

inline auto center_and_scale(vtkPolyData *poly) -> void {
  auto pts = get_points(poly);
  auto aabb = tf::aabb_from(tf::make_polygon(pts));
  auto center = aabb.center().as_vector();
  auto r = aabb.diagonal().length() / 2;
  tf::parallel_apply(pts.as_vector_view(), [&](auto pt) {
    pt -= center;
    pt *= 10 / r;
  });
}

template <typename Policy>
auto curves_to_polydata(const tf::curves<Policy> &curves) {
  auto cells = vtk_make_unique<vtkCellArray>();

  for (auto path : curves.paths()) {
    std::vector<vtkIdType> ids(path.begin(), path.end());
    cells->InsertNextCell(path.size(), ids.data());
  }
  auto points = vtk_make_unique<vtkPoints>();
  points->SetNumberOfPoints(curves.points().size());
  tf::parallel_copy(curves.points(), get_points(points.get()));
  auto tmp_poly = vtk_make_unique<vtkPolyData>();
  tmp_poly->SetPoints(points.get());
  tmp_poly->SetLines(cells.get());
  return tmp_poly;
}

template <typename Policy>
auto segments_to_lines(const tf::segments<Policy> &segments) {
  auto paths = tf::connect_edges_to_paths(segments.edges());
  return curves_to_polydata(tf::make_curves(paths, segments.points()));
}
