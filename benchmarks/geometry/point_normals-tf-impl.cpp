/**
 * Point normals benchmark with TrueForm - Implementation
 *
 * Topology is computed from scratch for each run to match libigl.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "point_normals-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_point_normals_tf_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto mesh = tf::read_stl<int>(path);
    auto polygons = mesh.polygons();

    tf::unit_vectors_buffer<float, 3> point_normals;

    auto time = benchmark::min_time_of(
        [&]() {
          point_normals = tf::compute_point_normals(polygons);
          benchmark::do_not_optimize(point_normals);
        },
        n_samples);

    out << polygons.size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
