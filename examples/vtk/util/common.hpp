#pragma once
#include "./data_bridge.hpp"
#include "vtkCamera.h"
#include "vtkCommand.h"
#include "vtkOBJReader.h"
#include "vtkPolyData.h"
#include "vtkProperty.h"
#include "vtkRenderWindow.h"
#include "vtkRenderer.h"
#include "vtkSTLReader.h"
#include "vtkStripper.h"
#include "vtkTextActor.h"
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

inline auto readOBJ(std::string name) {
  auto reader = vtk_make_unique<vtkOBJReader>();
  reader->SetFileName(name.c_str());
  reader->Update();
  auto out = vtk_make_unique<vtkPolyData>();
  out->ShallowCopy(reader->GetOutput());
  return out;
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
  auto cells = vtk_make_unique<vtkCellArray>();
  cells->AllocateEstimate(segments.size(), 2);
  for (auto [id0, id1] : segments.edges()) {
    vtkIdType ids[2]{id0, id1};
    cells->InsertNextCell(2, ids);
  }
  auto points = vtk_make_unique<vtkPoints>();
  points->SetNumberOfPoints(segments.points().size());
  tf::parallel_copy(segments.points(), get_points(points.get()));
  auto tmp_poly = vtk_make_unique<vtkPolyData>();
  tmp_poly->SetPoints(points.get());
  tmp_poly->SetLines(cells.get());
  auto stripper = vtk_make_unique<vtkStripper>();
  stripper->SetInputData(tmp_poly.get());
  stripper->Update();
  tmp_poly->ShallowCopy(stripper->GetOutput());
  return tmp_poly;
}
