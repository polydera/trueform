/**
 * Embedded self-intersection curves benchmark with libigl - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "conversions.hpp"
#include "embedded_self_intersection_curves-igl.hpp"
#include "rotation.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <igl/copyleft/cgal/remesh_self_intersections.h>

namespace benchmark {

int run_embedded_self_intersection_curves_igl_benchmark(
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

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
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
          V = benchmark::igl::to_igl_vertices(concatenated_mesh.points());
          F = benchmark::igl::to_igl_faces(concatenated_mesh.faces());
          ++iter;
        },
        [&]() {
          Eigen::MatrixXd VV;
          Eigen::MatrixXi FF;
          Eigen::MatrixXi IF;
          Eigen::VectorXi J;
          Eigen::VectorXi IM;
          ::igl::copyleft::cgal::remesh_self_intersections(V, F, {}, VV, FF, IF,
                                                           J, IM);
          benchmark::do_not_optimize(VV);
          benchmark::do_not_optimize(FF);
        },
        n_samples);

    out << r_polygons.faces().size() << "," << r_polygons.faces().size() << ","
        << time_ms << "\n";
  }

  return 0;
}

} // namespace benchmark
