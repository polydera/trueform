/**
 * Benchmark: Closest-point queries with trueform
 *
 * Measures time to perform k-nearest neighbor queries on point clouds
 * of varying sizes using TrueForm's tf::tree.
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "test_meshes.hpp"
#include "timing.hpp"
#include <iostream>
#include <trueform/trueform.hpp>

int main() {
  std::cout << "polygons,points,time_ms\n";

  constexpr int n_samples = 10;

  for (const auto &path : benchmark::BENCHMARK_MESHES) {
    auto polygons = tf::read_stl<int>(path);
    auto points = polygons.points();

    // Build tree
    tf::tree<int, float, 3> tree;
    tree.build(polygons.polygons(), tf::config_tree(4, 4));

    // Calculate diagonal length for query generation
    auto l = tf::aabb_from(points).diagonal().length();

    auto form_polygons = tf::make_form(tree, polygons.polygons());

    tf::tree<int, float, 3> point_tree;
    auto point_cloud = polygons.points_buffer(); // copy of buffer

    auto time_ms = benchmark::mean_time_of(
        [&]() {
          // Copy raw buffer (your suggestion)
          auto point_cloud = polygons.points_buffer(); // copy of buffer

          // Pick a random "pivot" point for the random transformation
          auto pivot_idx =
              tf::random<int>(0, static_cast<int>(point_cloud.size()));
          auto pivot = points[pivot_idx];

          auto transformation = tf::random_transformation_at(pivot);

          // Random translation: pick another random point from the cloud
          auto translation = tf::random_vector<float, 3>() * l;

          // Apply transform + translation to all copied points
          // point' = transformed(point, T) + translation.as_vector_view()
          for (auto &&p : point_cloud) {
            p = tf::transformed(p, transformation) + translation;
          }

          point_tree.build(point_cloud.points(), tf::config_tree(4, 4));

        },
        [&]() {
          // Benchmark: kNN query
          auto cpt = tf::neighbor_search(
              form_polygons, tf::make_form(point_tree, point_cloud.points()));
          benchmark::do_not_optimize(cpt);
        },
        n_samples);

    std::cout << polygons.faces().size() << "," << points.size() << ","
              << time_ms << "\n";
  }

  return 0;
}
