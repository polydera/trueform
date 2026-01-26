/**
 * Polygons to polygons collision TrueForm - Implementation
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "polygons_to_polygons-collision-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_polygons_to_polygons_collision_tf_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "bv,polygons,polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto polygons = tf::read_stl<int>(path);
    auto points = polygons.points();
    auto config = tf::config_tree(4, 4);
    auto l = tf::aabb_from(points).diagonal().length();
    tf::frame<float, 3> frame;

    tf::aabb_tree<int, float, 3> tree_aabb;
    tree_aabb.build(polygons.polygons(), config);
    auto form_aabb = polygons.polygons() | tf::tag(tree_aabb);

    auto time_aabb = benchmark::mean_time_of(
        [&]() {
          auto pivot_idx = tf::random<int>(0, static_cast<int>(points.size()) - 1);
          auto pivot = points[pivot_idx];
          auto translation = tf::random_vector<float, 3>() * 2 * l;
          frame = tf::make_frame(
              tf::random_transformation_at(pivot, pivot + translation));
        },
        [&]() {
          auto result = tf::intersects(
              form_aabb, polygons.polygons() | tf::tag(tree_aabb) | tf::tag(frame));
          benchmark::do_not_optimize(result);
        },
        n_samples);
    out << "AABB," << polygons.faces().size() << "," << polygons.size() << ","
        << time_aabb << "\n";

    tf::obb_tree<int, float, 3> tree_obb;
    tree_obb.build(polygons.polygons(), config);
    auto form_obb = polygons.polygons() | tf::tag(tree_obb);

    auto time_obb = benchmark::mean_time_of(
        [&]() {
          auto pivot_idx = tf::random<int>(0, static_cast<int>(points.size()) - 1);
          auto pivot = points[pivot_idx];
          auto translation = tf::random_vector<float, 3>() * 2 * l;
          frame = tf::make_frame(
              tf::random_transformation_at(pivot, pivot + translation));
        },
        [&]() {
          auto result = tf::intersects(
              form_obb, polygons.polygons() | tf::tag(tree_obb) | tf::tag(frame));
          benchmark::do_not_optimize(result);
        },
        n_samples);
    out << "OBB," << polygons.faces().size() << "," << polygons.size() << ","
        << time_obb << "\n";

    tf::obbrss_tree<int, float, 3> tree_obbrss;
    tree_obbrss.build(polygons.polygons(), config);
    auto form_obbrss = polygons.polygons() | tf::tag(tree_obbrss);

    auto time_obbrss = benchmark::mean_time_of(
        [&]() {
          auto pivot_idx = tf::random<int>(0, static_cast<int>(points.size()) - 1);
          auto pivot = points[pivot_idx];
          auto translation = tf::random_vector<float, 3>() * 2 * l;
          frame = tf::make_frame(
              tf::random_transformation_at(pivot, pivot + translation));
        },
        [&]() {
          auto result = tf::intersects(
              form_obbrss, polygons.polygons() | tf::tag(tree_obbrss) | tf::tag(frame));
          benchmark::do_not_optimize(result);
        },
        n_samples);
    out << "OBBRSS," << polygons.faces().size() << "," << polygons.size() << ","
        << time_obbrss << "\n";
  }

  return 0;
}

} // namespace benchmark
