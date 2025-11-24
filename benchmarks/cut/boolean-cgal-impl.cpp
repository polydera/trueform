/**
 * Boolean operations benchmark with CGAL - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "boolean-cgal.hpp"
#include "conversions.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <CGAL/Polygon_mesh_processing/corefinement.h>

namespace PMP = CGAL::Polygon_mesh_processing;
using namespace benchmark::cgal;

namespace benchmark {

int run_boolean_cgal_benchmark(const std::vector<std::string> &mesh_paths,
                               int n_samples, std::ostream &out) {
  out << "polygons0,polygons1,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);
    auto points = r_polygons.points();

    // Create mesh1
    auto mesh1 = benchmark::cgal::to_cgal_mesh_d(r_polygons);

    // Compute deterministic translation: 50% along largest axis
    auto aabb = tf::aabb_from(r_polygons.polygons());
    int axis = tf::largest_axis(aabb.diagonal());
    float offset = aabb.diagonal()[axis] * 0.5f;

    auto translation = tf::vector<float, 3>(0.0f, 0.0f, 0.0f);
    translation[axis] = offset;
    auto transform = tf::make_transformation_from_translation(translation);

    // Transform TrueForm mesh once, then convert to CGAL
    tf::polygons_buffer<int, float, 3, 3> transformed;
    transformed.faces_buffer() = r_polygons.faces_buffer();
    transformed.points_buffer().allocate(points.size());
    tf::parallel_transform(points, transformed.points(), [&](auto pt) {
      return tf::transformed(pt, transform);
    });

    auto mesh2 = benchmark::cgal::to_cgal_mesh_d(transformed);

    auto time_ms = benchmark::min_time_of(
        [&]() {
          Surface_mesh_d mesh1_copy = mesh1;
          Surface_mesh_d mesh2_copy = mesh2;
          Surface_mesh_d result;
          PMP::corefine_and_compute_union(mesh1_copy, mesh2_copy, result);
          benchmark::do_not_optimize(result);
        },
        n_samples);

    out << r_polygons.faces().size() << "," << r_polygons.faces().size() << ","
        << time_ms << "\n";
  }

  return 0;
}

} // namespace benchmark
