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
  out << "bv,polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);
    auto polygons = r_polygons.polygons();
    auto config = tf::config_tree(4, 4);

    auto time_aabb = benchmark::min_time_of(
        [&]() {
          tf::aabb_tree<int, float, 3> tree;
          tree.build(polygons, config);
          benchmark::do_not_optimize(tree);
        },
        n_samples);
    out << "AABB," << polygons.size() << "," << time_aabb << "\n";

    auto time_obb = benchmark::min_time_of(
        [&]() {
          tf::obb_tree<int, float, 3> tree;
          tree.build(polygons, config);
          benchmark::do_not_optimize(tree);
        },
        n_samples);
    out << "OBB," << polygons.size() << "," << time_obb << "\n";

    auto time_obbrss = benchmark::min_time_of(
        [&]() {
          tf::obbrss_tree<int, float, 3> tree;
          tree.build(polygons, config);
          benchmark::do_not_optimize(tree);
        },
        n_samples);
    out << "OBBRSS," << polygons.size() << "," << time_obbrss << "\n";
  }

  return 0;
}

} // namespace benchmark
