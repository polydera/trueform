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
  out << "bv,points,k,time_ms\n";

  constexpr int max_k = 10;

  for (const auto &path : mesh_paths) {
    auto polygons = tf::read_stl<int>(path);
    auto points = polygons.points();
    auto config = tf::config_tree(4, 4);
    auto l = tf::aabb_from(points).diagonal().length();
    tf::point<float, 3> query_point;
    std::array<tf::nearest_neighbor<int, float, 3>, max_k> buffer;

    tf::aabb_tree<int, float, 3> tree_aabb;
    tree_aabb.build(points, config);
    auto form_aabb = points | tf::tag(tree_aabb);

    for (int k = 1; k <= max_k; ++k) {
      auto time = benchmark::mean_time_of(
          [&]() {
            auto idx = tf::random<int>(0, points.size() - 1);
            query_point = points[idx] + tf::random_vector<float, 3>() * l;
          },
          [&]() {
            auto knn = tf::make_nearest_neighbors(buffer.begin(), k);
            tf::neighbor_search(form_aabb, query_point, knn);
            benchmark::do_not_optimize(knn);
          },
          n_samples);
      out << "AABB," << points.size() << "," << k << "," << time << "\n";
    }

    tf::obb_tree<int, float, 3> tree_obb;
    tree_obb.build(points, config);
    auto form_obb = points | tf::tag(tree_obb);

    for (int k = 1; k <= max_k; ++k) {
      auto time = benchmark::mean_time_of(
          [&]() {
            auto idx = tf::random<int>(0, points.size() - 1);
            query_point = points[idx] + tf::random_vector<float, 3>() * l;
          },
          [&]() {
            auto knn = tf::make_nearest_neighbors(buffer.begin(), k);
            tf::neighbor_search(form_obb, query_point, knn);
            benchmark::do_not_optimize(knn);
          },
          n_samples);
      out << "OBB," << points.size() << "," << k << "," << time << "\n";
    }

    tf::obbrss_tree<int, float, 3> tree_obbrss;
    tree_obbrss.build(points, config);
    auto form_obbrss = points | tf::tag(tree_obbrss);

    for (int k = 1; k <= max_k; ++k) {
      auto time = benchmark::mean_time_of(
          [&]() {
            auto idx = tf::random<int>(0, points.size() - 1);
            query_point = points[idx] + tf::random_vector<float, 3>() * l;
          },
          [&]() {
            auto knn = tf::make_nearest_neighbors(buffer.begin(), k);
            tf::neighbor_search(form_obbrss, query_point, knn);
            benchmark::do_not_optimize(knn);
          },
          n_samples);
      out << "OBBRSS," << points.size() << "," << k << "," << time << "\n";
    }
  }

  return 0;
}

} // namespace benchmark
