/**
 * Point cloud tree building benchmark with TrueForm - Implementation
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "point_cloud-build_tree-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_point_cloud_build_tree_tf_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "points,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto polygons = tf::read_stl<int>(path);
    auto points = polygons.points();

    auto time = benchmark::min_time_of(
        [&]() {
          tf::tree<int, float, 3> tree;
          tree.build(points, tf::config_tree(4, 4));
          benchmark::do_not_optimize(tree);
        },
        n_samples);

    out << points.size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
