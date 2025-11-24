/**
 * Polygons tree building TrueForm - Implementation
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "polygons-build_tree-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_polygons_build_tree_tf_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);
    auto polygons = r_polygons.polygons();

    auto time = benchmark::min_time_of(
        [&]() {
          tf::tree<int, float, 3> tree;
          tree.build(polygons, tf::config_tree(4, 4));
          benchmark::do_not_optimize(tree);
        },
        n_samples);

    out << polygons.size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
