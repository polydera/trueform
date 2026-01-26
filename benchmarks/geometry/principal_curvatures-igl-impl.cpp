/**
 * Principal curvatures benchmark with libigl - Implementation
 *
 * libigl crashes on benchmark meshes due to non-manifold geometry.
 * TrueForm handles them normally. Hence we generate manifold sphere
 * meshes sized to match each input mesh's polygon count for comparable
 * scaling behavior.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "principal_curvatures-igl.hpp"
#include "conversions.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <igl/principal_curvature.h>

#include <cmath>

namespace benchmark {

namespace {
// Compute sphere parameters to approximate target face count.
// Sphere faces = 2 * segments * (stacks - 1)
// Using stacks ≈ segments = s: faces ≈ 2 * s^2
std::pair<int, int> sphere_params_for_faces(std::size_t target_faces) {
  int s = static_cast<int>(std::sqrt(target_faces / 2.0)) + 1;
  if (s < 4)
    s = 4; // minimum for reasonable sphere
  return {s, s};
}
} // namespace

int run_principal_curvatures_igl_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto input_mesh = tf::read_stl<int>(path);
    const auto target_faces = input_mesh.faces().size();

    // Generate manifold sphere with similar polygon count
    auto [stacks, segments] = sphere_params_for_faces(target_faces);
    auto sphere = tf::make_sphere_mesh(1.0f, stacks, segments);

    // Convert to libigl format
    auto V = benchmark::igl::to_igl_vertices(sphere.points());
    auto F = benchmark::igl::to_igl_faces(sphere.faces());

    Eigen::MatrixXd PD1, PD2;
    Eigen::VectorXd PV1, PV2;

    auto time = benchmark::min_time_of(
        [&]() {
          // Use radius=2 to match TrueForm's default k=2
          ::igl::principal_curvature(V, F, PD1, PD2, PV1, PV2, 2, true);
          benchmark::do_not_optimize(PV1);
          benchmark::do_not_optimize(PV2);
        },
        n_samples);

    out << sphere.faces().size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
