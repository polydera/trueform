/**
 * ICP registration benchmark with TrueForm - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "icp_registration-tf.hpp"
#include "../common/timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_icp_registration_tf_benchmark(const std::vector<std::string> &mesh_paths,
                                      int n_samples, std::ostream &out) {
  out << "polygons,time_ms,chamfer_error\n";

  for (const auto &path : mesh_paths) {
    auto mesh = tf::read_stl<int>(path);
    auto polygons = mesh.polygons();
    const auto n_polys = polygons.size();

    // Build vertex link for Taubin smoothing
    auto vlink = tf::make_vertex_link(polygons);

    // Create target: Taubin smoothed version (200 iterations)
    auto target_points_buf =
        tf::taubin_smoothed(polygons.points() | tf::tag(vlink), 200, 0.5f, 0.1f);

    // Build target polygons with smoothed points
    tf::polygons_buffer<int, float, 3, 3> target_mesh;
    target_mesh.faces_buffer() = mesh.faces_buffer();
    target_mesh.points_buffer() = std::move(target_points_buf);
    auto target_polygons = target_mesh.polygons();

    // Build tree on target points (point-to-point ICP, no normals)
    tf::aabb_tree<int, float, 3> tree;
    tree.build(target_polygons.points(), tf::config_tree(4, 4));

    // Prepare target with tree only
    auto target = target_polygons.points() | tf::tag(tree);

    // Compute target centroid for center alignment
    auto target_center = tf::centroid(target_polygons.points());

    // ICP config - fixed 30 iterations, no early termination
    tf::icp_config config;
    config.max_iterations = 30;
    config.min_relative_improvement = 0;
    config.n_samples = 1000;
    config.k = 1;

    // Pre-generate all random angles
    std::vector<std::array<float, 3>> angles(n_samples);
    for (int i = 0; i < n_samples; ++i) {
      angles[i] = {tf::random<float>(-10.0f, 10.0f),
                   tf::random<float>(-10.0f, 10.0f),
                   tf::random<float>(-10.0f, 10.0f)};
    }

    // Store ICP results for chamfer computation
    std::vector<tf::frame<float, 3>> icp_results(n_samples);
    int iter = 0;

    // Current source points (set in prepare) - fresh copy like VTK/libigl
    auto center = tf::centroid(polygons.points());
    tf::points_buffer<float, 3> current_source;
    current_source.allocate(polygons.points().size());

    auto time = benchmark::mean_time_of(
        [&]() {
          // Prepare: build source points for this iteration (not timed)
          auto [ax, ay, az] = angles[iter];
          auto rx = tf::make_rotation(tf::deg(ax), tf::axis<0>, center);
          auto ry = tf::make_rotation(tf::deg(ay), tf::axis<1>, center);
          auto rz = tf::make_rotation(tf::deg(az), tf::axis<2>, center);
          auto rotation = tf::transformed(tf::transformed(rx, ry), rz);

          // Apply rotation and center alignment
          for (std::size_t i = 0; i < polygons.points().size(); ++i) {
            auto pt = tf::transformed(polygons.points()[i], rotation);
            current_source[i] = tf::make_point(
                pt[0] - center[0] + target_center[0],
                pt[1] - center[1] + target_center[1],
                pt[2] - center[2] + target_center[2]);
          }
        },
        [&]() {
          // Timed: run ICP
          auto T = tf::fit_icp_alignment(current_source.points(), target, config);
          icp_results[iter] = T;
          ++iter;
        },
        n_samples);

    // Compute mean chamfer error (relative to mesh diagonal)
    auto aabb = tf::aabb_from(polygons.points());
    float diagonal = aabb.diagonal().length();

    float total_chamfer = 0.0f;
    for (int i = 0; i < n_samples; ++i) {
      auto [ax, ay, az] = angles[i];
      auto rx = tf::make_rotation(tf::deg(ax), tf::axis<0>, center);
      auto ry = tf::make_rotation(tf::deg(ay), tf::axis<1>, center);
      auto rz = tf::make_rotation(tf::deg(az), tf::axis<2>, center);
      auto rotation = tf::transformed(tf::transformed(rx, ry), rz);

      // Rebuild source points
      tf::points_buffer<float, 3> source_pts;
      source_pts.allocate(polygons.points().size());
      for (std::size_t j = 0; j < polygons.points().size(); ++j) {
        auto pt = tf::transformed(polygons.points()[j], rotation);
        source_pts[j] = tf::make_point(
            pt[0] - center[0] + target_center[0],
            pt[1] - center[1] + target_center[1],
            pt[2] - center[2] + target_center[2]);
      }

      auto aligned = source_pts.points() | tf::tag(icp_results[i]);
      total_chamfer += tf::chamfer_error(aligned, target) / diagonal;
    }
    float mean_chamfer = total_chamfer / n_samples;

    out << n_polys << "," << time << "," << mean_chamfer << "\n";
  }

  return 0;
}

} // namespace benchmark
