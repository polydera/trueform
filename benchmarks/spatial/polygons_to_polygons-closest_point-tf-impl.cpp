/**
 * Polygons to polygons closest-point TrueForm - Implementation
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "polygons_to_polygons-closest_point-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_polygons_to_polygons_closest_point_tf_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto polygons = tf::read_stl<int>(path);
    auto points = polygons.points();

    // Build tree
    tf::aabb_tree<int, float, 3> tree;
    tree.build(polygons.polygons(), tf::config_tree(4, 4));

    // Calculate diagonal length for query generation
    auto l = tf::aabb_from(points).diagonal().length();

    auto form_polygons = tf::make_form(tree, polygons.polygons());
    tf::aabb_tree<int, float, 3> tree1;
    auto polygons1 = polygons;

    tf::frame<float, 3> frame;
    auto time_ms = benchmark::mean_time_of(
        [&]() {
          // Pick a random "pivot" point for the random transformation
          auto pivot_idx = tf::random<int>(0, static_cast<int>(points.size()) - 1);
          auto pivot = points[pivot_idx];

          auto translation = tf::random_vector<float, 3>() * 2 * l;
          frame = tf::make_frame(
              tf::random_transformation_at(pivot, pivot + translation));
        },
        [&]() {
          // Benchmark: kNN query
          auto cpt = tf::neighbor_search(
              form_polygons, tf::make_form(frame, tree, polygons.polygons()));
          benchmark::do_not_optimize(cpt);
        },
        n_samples);

    out << polygons.faces().size() << "," << polygons.size() << "," << time_ms
        << "\n";
  }

  return 0;
}

} // namespace benchmark
