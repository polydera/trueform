/**
 * Point cloud kNN queries benchmark with TrueForm - Implementation
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "point_cloud-knn-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_point_cloud_knn_tf_benchmark(const std::vector<std::string> &mesh_paths,
                                     int n_samples, std::ostream &out) {
  out << "points,k,time_ms\n";

  constexpr int max_k = 10;

  for (const auto &path : mesh_paths) {
    auto polygons = tf::read_stl<int>(path);
    auto points = polygons.points();

    // Build tree
    tf::tree<int, float, 3> tree;
    tree.build(points, tf::config_tree(4, 4));

    // Calculate diagonal length for query generation
    auto l = tf::aabb_from(points).diagonal().length();

    // Query point (updated in prepare)
    tf::point<float, 3> query_point;
    auto form = tf::make_form(tree, points);

    // Buffer for kNN results
    std::array<tf::nearest_neighbor<int, float, 3>, max_k> buffer;

    for (int k = 1; k <= max_k; ++k) {
      auto time = benchmark::mean_time_of(
          [&]() {
            // Prepare: generate random query point
            auto idx = tf::random<int>(0, points.size() - 1);
            auto point = points[idx];
            query_point = point + tf::random_vector<float, 3>() * l;
          },
          [&]() {
            // Benchmark: kNN query
            auto knn = tf::make_nearest_neighbors(buffer.begin(), k);
            tf::neighbor_search(form, query_point, knn);
            benchmark::do_not_optimize(knn);
          },
          n_samples);

      out << points.size() << "," << k << "," << time << "\n";
    }
  }

  return 0;
}

} // namespace benchmark
