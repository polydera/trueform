/**
 * ICP registration benchmark with VTK - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "icp_registration-vtk.hpp"
#include "../common/timing.hpp"
#include "../vtk-common/conversions.hpp"
#include <trueform/trueform.hpp>

#include <vtkIterativeClosestPointTransform.h>
#include <vtkLandmarkTransform.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

namespace benchmark {

int run_icp_registration_vtk_benchmark(const std::vector<std::string> &mesh_paths,
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

    // Convert target to VTK
    auto target_vtk = benchmark::vtk::to_vtk_polydata(target_mesh);

    // Build tree on target points for chamfer computation
    tf::aabb_tree<int, float, 3> tree;
    tree.build(target_polygons.points(), tf::config_tree(4, 4));
    auto target = target_polygons.points() | tf::tag(tree);

    // Pre-generate all random angles
    std::vector<std::array<float, 3>> angles(n_samples);
    for (int i = 0; i < n_samples; ++i) {
      angles[i] = {tf::random<float>(-10.0f, 10.0f),
                   tf::random<float>(-10.0f, 10.0f),
                   tf::random<float>(-10.0f, 10.0f)};
    }

    // Store ICP results as 4x4 matrices for chamfer computation
    std::vector<std::array<double, 16>> icp_results(n_samples);
    int iter = 0;

    // Current source (set in prepare)
    vtkSmartPointer<vtkPolyData> current_source_vtk;
    auto center = tf::centroid(polygons.points());
    auto target_center = tf::centroid(target_polygons.points());

    auto time = benchmark::mean_time_of(
        [&]() {
          // Prepare: build source mesh for this iteration (not timed)
          auto [ax, ay, az] = angles[iter];
          auto rx = tf::make_rotation(tf::deg(ax), tf::axis<0>, center);
          auto ry = tf::make_rotation(tf::deg(ay), tf::axis<1>, center);
          auto rz = tf::make_rotation(tf::deg(az), tf::axis<2>, center);
          auto rotation = tf::transformed(tf::transformed(rx, ry), rz);

          tf::polygons_buffer<int, float, 3, 3> source_mesh;
          source_mesh.faces_buffer() = mesh.faces_buffer();
          source_mesh.points_buffer().allocate(polygons.points().size());
          for (std::size_t i = 0; i < polygons.points().size(); ++i) {
            auto pt = tf::transformed(polygons.points()[i], rotation);
            source_mesh.points_buffer()[i] = pt;
          }
          current_source_vtk = benchmark::vtk::to_vtk_polydata(source_mesh);
        },
        [&]() {
          // Timed: run VTK ICP
          auto icp = vtkSmartPointer<vtkIterativeClosestPointTransform>::New();
          icp->SetSource(current_source_vtk);
          icp->SetTarget(target_vtk);
          icp->GetLandmarkTransform()->SetModeToRigidBody();
          icp->StartByMatchingCentroidsOn();
          icp->SetMaximumNumberOfIterations(30);
          icp->SetMaximumNumberOfLandmarks(1000);
          icp->CheckMeanDistanceOff();
          icp->Update();

          // Store 4x4 matrix elements row-major
          auto *m = icp->GetMatrix();
          for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
              icp_results[iter][r * 4 + c] = m->GetElement(r, c);
            }
          }
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

      // VTK does center alignment, so rebuild centered source
      tf::points_buffer<float, 3> rotated_pts;
      rotated_pts.allocate(polygons.points().size());
      for (std::size_t j = 0; j < polygons.points().size(); ++j) {
        rotated_pts[j] = tf::transformed(polygons.points()[j], rotation);
      }
      auto source_center = tf::centroid(rotated_pts.points());

      // Apply center alignment + ICP transform: p' = M * (p + center_offset)
      const auto &M = icp_results[i];
      tf::points_buffer<float, 3> aligned_pts;
      aligned_pts.allocate(rotated_pts.size());
      for (std::size_t j = 0; j < rotated_pts.size(); ++j) {
        // Center-aligned point
        float px = rotated_pts[j][0] - source_center[0] + target_center[0];
        float py = rotated_pts[j][1] - source_center[1] + target_center[1];
        float pz = rotated_pts[j][2] - source_center[2] + target_center[2];
        // Apply 4x4 transform
        aligned_pts[j] = tf::make_point(
            float(M[0] * px + M[1] * py + M[2] * pz + M[3]),
            float(M[4] * px + M[5] * py + M[6] * pz + M[7]),
            float(M[8] * px + M[9] * py + M[10] * pz + M[11]));
      }
      total_chamfer += tf::chamfer_error(aligned_pts.points(), target) / diagonal;
    }
    float mean_chamfer = total_chamfer / n_samples;

    out << n_polys << "," << time << "," << mean_chamfer << "\n";
  }

  return 0;
}

} // namespace benchmark
