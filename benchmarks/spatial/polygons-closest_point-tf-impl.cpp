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
  out << "bv,polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto polygons = tf::read_stl<int>(path);
    auto points = polygons.points();
    auto config = tf::config_tree(4, 4);
    auto l = tf::aabb_from(points).diagonal().length();
    tf::point<float, 3> query_point;

    tf::aabb_tree<int, float, 3> tree_aabb;
    tree_aabb.build(polygons.polygons(), config);
    auto form_aabb = polygons.polygons() | tf::tag(tree_aabb);

    auto time_aabb = benchmark::mean_time_of(
        [&]() {
          auto idx = tf::random<int>(0, points.size() - 1);
          query_point = points[idx] + tf::random_vector<float, 3>() * l;
        },
        [&]() {
          auto cpt = tf::neighbor_search(form_aabb, query_point);
          benchmark::do_not_optimize(cpt);
        },
        n_samples);
    out << "AABB," << polygons.faces().size() << "," << time_aabb << "\n";

    tf::obb_tree<int, float, 3> tree_obb;
    tree_obb.build(polygons.polygons(), config);
    auto form_obb = polygons.polygons() | tf::tag(tree_obb);

    auto time_obb = benchmark::mean_time_of(
        [&]() {
          auto idx = tf::random<int>(0, points.size() - 1);
          query_point = points[idx] + tf::random_vector<float, 3>() * l;
        },
        [&]() {
          auto cpt = tf::neighbor_search(form_obb, query_point);
          benchmark::do_not_optimize(cpt);
        },
        n_samples);
    out << "OBB," << polygons.faces().size() << "," << time_obb << "\n";

    tf::obbrss_tree<int, float, 3> tree_obbrss;
    tree_obbrss.build(polygons.polygons(), config);
    auto form_obbrss = polygons.polygons() | tf::tag(tree_obbrss);

    auto time_obbrss = benchmark::mean_time_of(
        [&]() {
          auto idx = tf::random<int>(0, points.size() - 1);
          query_point = points[idx] + tf::random_vector<float, 3>() * l;
        },
        [&]() {
          auto cpt = tf::neighbor_search(form_obbrss, query_point);
          benchmark::do_not_optimize(cpt);
        },
        n_samples);
    out << "OBBRSS," << polygons.faces().size() << "," << time_obbrss << "\n";
  }

  return 0;
}

} // namespace benchmark
