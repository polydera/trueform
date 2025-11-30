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
  out << "bv,points,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto polygons = tf::read_stl<int>(path);
    auto points = polygons.points();
    auto config = tf::config_tree(4, 4);

    auto time_aabb = benchmark::min_time_of(
        [&]() {
          tf::aabb_tree<int, float, 3> tree;
          tree.build(points, config);
          benchmark::do_not_optimize(tree);
        },
        n_samples);
    out << "AABB," << points.size() << "," << time_aabb << "\n";

    auto time_obb = benchmark::min_time_of(
        [&]() {
          tf::obb_tree<int, float, 3> tree;
          tree.build(points, config);
          benchmark::do_not_optimize(tree);
        },
        n_samples);
    out << "OBB," << points.size() << "," << time_obb << "\n";

    auto time_obbrss = benchmark::min_time_of(
        [&]() {
          tf::obbrss_tree<int, float, 3> tree;
          tree.build(points, config);
          benchmark::do_not_optimize(tree);
        },
        n_samples);
    out << "OBBRSS," << points.size() << "," << time_obbrss << "\n";
  }

  return 0;
}

} // namespace benchmark
