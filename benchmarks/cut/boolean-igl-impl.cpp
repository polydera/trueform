/**
 * Boolean operations benchmark with libigl - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "boolean-igl.hpp"
#include "conversions.hpp"
#include "rotation.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <igl/copyleft/cgal/mesh_boolean.h>

namespace benchmark {

int run_boolean_igl_benchmark(const std::vector<std::string> &mesh_paths,
                              int n_samples, std::ostream &out) {
  out << "polygons0,polygons1,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);
    auto points = r_polygons.points();

    // Convert to libigl format
    auto V1 = benchmark::igl::to_igl_vertices(r_polygons.points());
    auto F1 = benchmark::igl::to_igl_faces(r_polygons.faces());

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

    Eigen::MatrixXd V2;
    Eigen::MatrixXi F2;
    int iter = 0;

    auto time_ms = benchmark::mean_time_of(
        [&]() {
          auto angle =
              tf::deg<float>{360.0f * (iter + 0.51f) / float(n_samples)};
          auto rotation = benchmark::make_rotation(angle, rot_axis, pivot);
          tf::parallel_transform(points, transformed.points(), [&](auto pt) {
            return tf::transformed(pt, rotation);
          });
          V2 = benchmark::igl::to_igl_vertices(transformed.points());
          F2 = benchmark::igl::to_igl_faces(transformed.faces());
          ++iter;
        },
        [&]() {
          Eigen::MatrixXd VC;
          Eigen::MatrixXi FC;
          ::igl::copyleft::cgal::mesh_boolean(
              V1, F1, V2, F2, ::igl::MESH_BOOLEAN_TYPE_UNION, VC, FC);
          benchmark::do_not_optimize(VC);
          benchmark::do_not_optimize(FC);
        },
        n_samples);

    out << r_polygons.faces().size() << "," << r_polygons.faces().size() << ","
        << time_ms << "\n";
  }

  return 0;
}

} // namespace benchmark
