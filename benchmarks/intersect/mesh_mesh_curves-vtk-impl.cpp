/**
 * Mesh-mesh intersection curves benchmark with VTK - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "conversions.hpp"
#include "mesh_mesh_curves-vtk.hpp"
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

    // Compute deterministic translation: 50% along largest axis
    auto aabb = tf::aabb_from(r_polygons.polygons());
    int axis = tf::largest_axis(aabb.diagonal());
    float offset = aabb.diagonal()[axis] * 0.5f;

    auto translation = tf::vector<float, 3>(0.0f, 0.0f, 0.0f);
    translation[axis] = offset;
    auto transform = tf::make_transformation_from_translation(translation);

    // Transform TrueForm mesh once, then convert to VTK
    tf::polygons_buffer<int, float, 3, 3> transformed;
    transformed.faces_buffer() = r_polygons.faces_buffer();
    transformed.points_buffer().allocate(points.size());
    tf::parallel_transform(points, transformed.points(), [&](auto pt) {
      return tf::transformed(pt, transform);
    });

    auto mesh2 = benchmark::vtk::to_vtk_polydata(transformed);

    auto time_ms = benchmark::min_time_of(
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
