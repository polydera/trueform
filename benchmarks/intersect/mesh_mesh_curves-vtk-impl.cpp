/**
 * Mesh-mesh intersection curves benchmark with VTK - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "conversions.hpp"
#include "mesh_mesh_curves-vtk.hpp"
#include "rotation.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <vtkIntersectionPolyDataFilter.h>
#include <vtkSmartPointer.h>

namespace benchmark {

int run_mesh_mesh_curves_vtk_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons0,polygons1,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);
    auto points = r_polygons.points();

    // Create mesh1
    auto mesh1 = benchmark::vtk::to_vtk_polydata(r_polygons);

    // Rotation around centroid, using smallest axis
    auto aabb = tf::aabb_from(r_polygons.polygons());
    auto pivot = tf::centroid(r_polygons.polygons());
    auto diag = aabb.diagonal();
    auto inv_diag =
        tf::vector<float, 3>{1.0f / diag[0], 1.0f / diag[1], 1.0f / diag[2]};
    int rot_axis = tf::largest_axis(inv_diag);

    // Pre-allocate transformed mesh buffer
    tf::polygons_buffer<int, float, 3, 3> transformed;
    transformed.faces_buffer() = r_polygons.faces_buffer();
    transformed.points_buffer().allocate(points.size());

    vtkSmartPointer<vtkPolyData> mesh2;
    int iter = 0;

    auto time_ms = benchmark::mean_time_of(
        [&]() {
          auto angle =
              tf::deg<float>{360.0f * (iter + 0.5f) / float(n_samples)};
          auto rotation = benchmark::make_rotation(angle, rot_axis, pivot);
          tf::parallel_transform(points, transformed.points(), [&](auto pt) {
            return tf::transformed(pt, rotation);
          });
          mesh2 = benchmark::vtk::to_vtk_polydata(transformed);
          ++iter;
        },
        [&]() {
          auto intersectionFilter =
              vtkSmartPointer<vtkIntersectionPolyDataFilter>::New();
          intersectionFilter->SetInputData(0, mesh1);
          intersectionFilter->SetInputData(1, mesh2);
          intersectionFilter->Update();
          benchmark::do_not_optimize(intersectionFilter->GetOutput());
        },
        n_samples);

    out << r_polygons.faces().size() << "," << r_polygons.faces().size() << ","
        << time_ms << "\n";
  }

  return 0;
}

} // namespace benchmark
