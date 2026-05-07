/**
 * Boolean operations benchmark with Geogram - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "boolean-geogram.hpp"
#include "conversions.hpp"
#include "rotation.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_surface_intersection.h>

namespace benchmark {

int run_boolean_geogram_benchmark(const std::vector<std::string> &mesh_paths,
                                  int n_samples, std::ostream &out) {
  out << "polygons0,polygons1,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);
    auto points = r_polygons.points();

    GEO::Mesh mesh1;
    benchmark::geogram::to_geogram_mesh(r_polygons, mesh1);

    auto aabb = tf::aabb_from(r_polygons.polygons());
    auto pivot = tf::centroid(r_polygons.polygons());
    auto diag = aabb.diagonal();
    auto inv_diag =
        tf::vector<float, 3>{1.0f / diag[0], 1.0f / diag[1], 1.0f / diag[2]};
    int rot_axis = tf::largest_axis(inv_diag);

    tf::polygons_buffer<int, float, 3, 3> transformed;
    transformed.faces_buffer() = r_polygons.faces_buffer();
    transformed.points_buffer().allocate(points.size());

    GEO::Mesh mesh2;
    int iter = 0;

    auto time_ms = benchmark::mean_time_of(
        [&]() {
          auto angle =
              tf::deg<float>{360.0f * (iter + 0.5f) / float(n_samples)};
          auto rotation = benchmark::make_rotation(angle, rot_axis, pivot);
          tf::parallel_transform(points, transformed.points(), [&](auto pt) {
            return tf::transformed(pt, rotation);
          });
          benchmark::geogram::to_geogram_mesh(transformed, mesh2);
          ++iter;
        },
        [&]() {
          GEO::Mesh result;
          GEO::mesh_union(result, mesh1, mesh2);
          benchmark::do_not_optimize(result);
        },
        n_samples);

    out << r_polygons.faces().size() << "," << r_polygons.faces().size() << ","
        << time_ms << "\n";
  }

  return 0;
}

} // namespace benchmark
