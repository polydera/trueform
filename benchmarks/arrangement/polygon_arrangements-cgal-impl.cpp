/**
 * Polygon arrangements benchmark with CGAL - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "polygon_arrangements-cgal.hpp"
#include "conversions.hpp"
#include "rotation.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <CGAL/Polygon_mesh_processing/autorefinement.h>

namespace PMP = CGAL::Polygon_mesh_processing;
using namespace benchmark::cgal;

namespace benchmark {

int run_polygon_arrangements_cgal_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons0,polygons1,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);
    auto points = r_polygons.points();

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

    std::vector<Point_3_d> soup_points;
    std::vector<std::array<std::size_t, 3>> soup_triangles;
    int iter = 0;

    auto time_ms = benchmark::mean_time_of(
        [&]() {
          auto angle =
              tf::deg<float>{360.0f * (iter + 0.5f) / float(n_samples)};
          auto rotation = benchmark::make_rotation(angle, rot_axis, pivot);
          tf::parallel_transform(points, transformed.points(), [&](auto pt) {
            return tf::transformed(pt, rotation);
          });
          auto concatenated_mesh =
              tf::concatenated(r_polygons.polygons(), transformed.polygons());
          soup_points.clear();
          soup_points.reserve(concatenated_mesh.points().size());
          for (const auto &pt : concatenated_mesh.points())
            soup_points.emplace_back(pt[0], pt[1], pt[2]);
          soup_triangles = to_cgal_faces(concatenated_mesh.faces());
          ++iter;
        },
        [&]() {
          auto pts = soup_points;
          auto tris = soup_triangles;
          PMP::autorefine_triangle_soup(pts, tris);
          benchmark::do_not_optimize(pts);
          benchmark::do_not_optimize(tris);
        },
        n_samples);

    out << r_polygons.faces().size() << "," << r_polygons.faces().size() << ","
        << time_ms << "\n";
  }

  return 0;
}

} // namespace benchmark
