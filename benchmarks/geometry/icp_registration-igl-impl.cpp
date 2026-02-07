/**
 * ICP registration benchmark with libigl - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "icp_registration-igl.hpp"
#include "../common/timing.hpp"
#include "../igl-common/conversions.hpp"
#include <trueform/trueform.hpp>

#include <igl/iterative_closest_point.h>

namespace benchmark {

int run_icp_registration_igl_benchmark(const std::vector<std::string> &mesh_paths,
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

    // Build tree on target for chamfer computation
    tf::aabb_tree<int, float, 3> tree;
    tree.build(target_polygons.points(), tf::config_tree(4, 4));
    auto target_tf = target_polygons.points() | tf::tag(tree);

    // Convert target to libigl format
    Eigen::MatrixXd VY = benchmark::igl::to_igl_vertices(target_polygons.points());
    Eigen::MatrixXi FY = benchmark::igl::to_igl_faces(target_polygons.faces());

    // Pre-convert original mesh faces (don't change per iteration)
    Eigen::MatrixXi FX = benchmark::igl::to_igl_faces(polygons.faces());

    // Compute target centroid for center alignment
    Eigen::RowVector3d target_centroid = VY.colwise().mean();

    // Pre-generate all random angles
    std::vector<std::array<float, 3>> angles(n_samples);
    for (int i = 0; i < n_samples; ++i) {
      angles[i] = {tf::random<float>(-10.0f, 10.0f),
                   tf::random<float>(-10.0f, 10.0f),
                   tf::random<float>(-10.0f, 10.0f)};
    }

    // Store ICP results for chamfer computation
    std::vector<std::pair<Eigen::Matrix3d, Eigen::RowVector3d>> icp_results(
        n_samples);
    // Store source centroids for chamfer computation
    std::vector<Eigen::RowVector3d> source_centroids(n_samples);
    int iter = 0;

    // Current source vertices (set in prepare)
    Eigen::MatrixXd current_VX;
    auto center = tf::centroid(polygons.points());

    auto time = benchmark::mean_time_of(
        [&]() {
          // Prepare: build source for this iteration (not timed)
          auto [ax, ay, az] = angles[iter];
          auto rx = tf::make_rotation(tf::deg(ax), tf::axis<0>, center);
          auto ry = tf::make_rotation(tf::deg(ay), tf::axis<1>, center);
          auto rz = tf::make_rotation(tf::deg(az), tf::axis<2>, center);
          auto rotation = tf::transformed(tf::transformed(rx, ry), rz);

          current_VX.resize(polygons.points().size(), 3);
          for (std::size_t i = 0; i < polygons.points().size(); ++i) {
            auto pt = tf::transformed(polygons.points()[i], rotation);
            current_VX(i, 0) = pt[0];
            current_VX(i, 1) = pt[1];
            current_VX(i, 2) = pt[2];
          }

          // Center-align source to target
          source_centroids[iter] = current_VX.colwise().mean();
          current_VX.rowwise() -= source_centroids[iter];
          current_VX.rowwise() += target_centroid;
        },
        [&]() {
          // Timed: run libigl ICP
          Eigen::Matrix3d R;
          Eigen::RowVector3d t;
          ::igl::iterative_closest_point(current_VX, FX, VY, FY, 1000, 30, R,
                                         t);
          icp_results[iter] = {R, t};
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

      // Rebuild centered source
      tf::points_buffer<float, 3> source_pts;
      source_pts.allocate(polygons.points().size());
      for (std::size_t j = 0; j < polygons.points().size(); ++j) {
        auto pt = tf::transformed(polygons.points()[j], rotation);
        // Apply center alignment
        source_pts[j] = tf::make_point(
            float(pt[0] - source_centroids[i](0) + target_centroid(0)),
            float(pt[1] - source_centroids[i](1) + target_centroid(1)),
            float(pt[2] - source_centroids[i](2) + target_centroid(2)));
      }

      // Apply ICP result (R, t): p' = p * R + t
      auto [R, t] = icp_results[i];
      tf::points_buffer<float, 3> aligned_pts;
      aligned_pts.allocate(source_pts.size());
      for (std::size_t j = 0; j < source_pts.size(); ++j) {
        auto p = source_pts[j];
        aligned_pts[j] = tf::make_point(
            float(p[0] * R(0, 0) + p[1] * R(1, 0) + p[2] * R(2, 0) + t(0)),
            float(p[0] * R(0, 1) + p[1] * R(1, 1) + p[2] * R(2, 1) + t(1)),
            float(p[0] * R(0, 2) + p[1] * R(1, 2) + p[2] * R(2, 2) + t(2)));
      }

      total_chamfer += tf::chamfer_error(aligned_pts.points(), target_tf) / diagonal;
    }
    float mean_chamfer = total_chamfer / n_samples;

    out << n_polys << "," << time << "," << mean_chamfer << "\n";
  }

  return 0;
}

} // namespace benchmark
