/**
 * Embedded self-intersection curves benchmark with TrueForm - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "embedded_self_intersection_curves-tf.hpp"
#include "rotation.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_embedded_self_intersection_curves_tf_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons0,polygons1,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto mesh = tf::read_stl<int>(path);
    auto polygons = mesh.polygons();
    auto points = mesh.points();

    // Rotation around centroid, using smallest axis
    auto aabb = tf::aabb_from(polygons);
    auto pivot = tf::centroid(polygons);
    auto diag = aabb.diagonal();
    auto inv_diag =
        tf::vector<float, 3>{1.0f / diag[0], 1.0f / diag[1], 1.0f / diag[2]};
    int rot_axis = tf::largest_axis(inv_diag);

    // Pre-allocate transformed mesh buffer
    tf::polygons_buffer<int, float, 3, 3> transformed;
    transformed.faces_buffer() = mesh.faces_buffer();
    transformed.points_buffer().allocate(points.size());

    int iter = 0;

    auto time_ms = benchmark::mean_time_of(
        [&]() {
          auto angle =
              tf::deg<float>{360.0f * (iter + 0.5f) / float(n_samples)};
          auto rotation = benchmark::make_rotation(angle, rot_axis, pivot);
          tf::parallel_transform(points, transformed.points(), [&](auto pt) {
            return tf::transformed(pt, rotation);
          });
          ++iter;
        },
        [&]() {
          auto concatenated_mesh =
              tf::concatenated(polygons, transformed.polygons());
          auto result = tf::embedded_self_intersection_curves(
              concatenated_mesh.polygons());
          benchmark::do_not_optimize(result);
        },
        n_samples);

    out << polygons.size() << "," << polygons.size() << "," << time_ms << "\n";
  }

  return 0;
}

} // namespace benchmark
