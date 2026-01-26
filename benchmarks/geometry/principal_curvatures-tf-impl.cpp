/**
 * Principal curvatures benchmark with TrueForm - Implementation
 *
 * libigl crashes on benchmark meshes due to non-manifold geometry.
 * TrueForm handles them normally. Hence we generate manifold sphere
 * meshes sized to match each input mesh's polygon count for comparable
 * scaling behavior.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "principal_curvatures-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

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

int run_principal_curvatures_tf_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto input_mesh = tf::read_stl<int>(path);
    const auto target_faces = input_mesh.faces().size();

    // Generate manifold sphere with similar polygon count
    auto [stacks, segments] = sphere_params_for_faces(target_faces);
    auto sphere = tf::make_sphere_mesh(1.0f, stacks, segments);
    auto polygons = sphere.polygons();

    const auto n_vertices = polygons.points().size();
    tf::buffer<float> k0, k1;
    k0.allocate(n_vertices);
    k1.allocate(n_vertices);

    tf::unit_vectors_buffer<float, 3> d0, d1;
    d0.allocate(n_vertices);
    d1.allocate(n_vertices);

    auto time = benchmark::min_time_of(
        [&]() {
          // Use k=2 to match libigl's radius=2
          tf::compute_principal_curvatures(polygons, k0, k1, d0, d1, 2);
          benchmark::do_not_optimize(k0);
          benchmark::do_not_optimize(k1);
          benchmark::do_not_optimize(d0);
          benchmark::do_not_optimize(d1);
        },
        n_samples);

    out << polygons.size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
