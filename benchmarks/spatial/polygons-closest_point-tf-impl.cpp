/**
 * Polygons closest-point TrueForm - Implementation
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "polygons-closest_point-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_polygons_closest_point_tf_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto polygons = tf::read_stl<int>(path);
    auto points = polygons.points();

    // Build tree
    tf::tree<int, float, 3> tree;
    tree.build(polygons.polygons(), tf::config_tree(4, 4));

    // Calculate diagonal length for query generation
    auto l = tf::aabb_from(points).diagonal().length();

    // Query point (updated in prepare)
    tf::point<float, 3> query_point;
    auto form = tf::make_form(tree, polygons.polygons());

    auto time = benchmark::mean_time_of(
        [&]() {
          // Prepare: generate random query point
          auto idx = tf::random<int>(0, points.size() - 1);
          auto point = points[idx];
          query_point = point + tf::random_vector<float, 3>() * l;
        },
        [&]() {
          // Benchmark: kNN query
          auto cpt = tf::neighbor_search(form, query_point);
          benchmark::do_not_optimize(cpt);
        },
        n_samples);

    out << polygons.faces().size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
