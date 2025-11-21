/**
 * Benchmark: Boolean operations with TrueForm
 *
 * Measures time to compute boolean union between two meshes
 * using TrueForm's make_boolean.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "test_meshes.hpp"
#include "timing.hpp"
#include <iostream>
#include <trueform/trueform.hpp>

int main() {
  std::cout << "polygons0,polygons1,time_ms\n";

  constexpr int n_samples = 10;

  for (const auto &path : benchmark::BENCHMARK_MESHES) {
    auto mesh = tf::read_stl<int>(path);
    auto polygons = mesh.polygons();
    auto points = mesh.points();

    tf::face_membership<int> fm;
    fm.build(polygons);
    tf::manifold_edge_link<int, 3> mel;
    mel.build(polygons.faces(), fm);
    tf::tree<int, float, 3> tree(polygons, tf::config_tree(4, 4));

    auto form0 = tf::make_form(tree, polygons | tf::tag(mel) | tf::tag(fm));

    tf::polygons_buffer<int, float, 3, 3> transformed;
    tf::frame<float, 3> frame;

    auto time_ms = benchmark::mean_time_of(
        [&]() {
          // Pick a random "pivot" point for the random transformation
          auto pivot_idx = tf::random<int>(0, static_cast<int>(points.size()));
          auto pivot = points[pivot_idx];
          auto new_origin_id =
              tf::random<int>(0, static_cast<int>(points.size()));
          auto new_origin = points[new_origin_id];

          frame =
              tf::make_frame(tf::random_transformation_at(pivot, new_origin));
        },
        [&]() {
          auto form1 =
              tf::make_form(frame, tree, polygons | tf::tag(mel) | tf::tag(fm));
          auto result = tf::make_boolean(form0, form1, tf::boolean_op::merge);
          benchmark::do_not_optimize(result);
        },
        n_samples);

    std::cout << polygons.size() << "," << polygons.size() << "," << time_ms << "\n";
  }

  return 0;
}
