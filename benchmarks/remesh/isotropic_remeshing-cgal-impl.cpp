/**
 * Isotropic remeshing benchmark with CGAL - Implementation
 *
 * Remeshes each mesh with target_length = 2× mean_edge_length,
 * 3 iterations. CGAL's Surface_mesh includes built-in half-edge
 * connectivity, matching TrueForm's precomputed half-edges.
 * Then sweeps multipliers 0.5×–4× on the 1M mesh.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "isotropic_remeshing-cgal.hpp"
#include "conversions.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <CGAL/Polygon_mesh_processing/remesh.h>

namespace benchmark {

namespace PMP = CGAL::Polygon_mesh_processing;

int run_isotropic_remeshing_cgal_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,multiplier,time_ms\n";

  // All mesh sizes at 2× mel
  for (const auto &path : mesh_paths) {
    auto mesh = tf::read_stl<int>(path);
    float mel = tf::mean_edge_length(mesh.polygons());
    auto base_mesh = benchmark::cgal::to_cgal_mesh(mesh);

    cgal::Surface_mesh cgal_mesh;

    auto time = benchmark::min_time_of(
        [&]() { cgal_mesh = base_mesh; },
        [&]() {
          PMP::isotropic_remeshing(faces(cgal_mesh), 2.0f * mel, cgal_mesh,
              CGAL::parameters::number_of_iterations(3));
          benchmark::do_not_optimize(cgal_mesh);
        },
        n_samples);

    out << mesh.faces().size() << ",2," << time << "\n";
  }

  // 1M mesh at multipliers 0.5–4×
  auto mesh_1m = tf::read_stl<int>(mesh_paths.back());
  float mel_1m = tf::mean_edge_length(mesh_1m.polygons());
  auto base_1m = benchmark::cgal::to_cgal_mesh(mesh_1m);

  float multipliers[] = {1.0f, 2.0f, 3.0f, 4.0f};
  for (float mult : multipliers) {
    cgal::Surface_mesh cgal_mesh;

    auto time = benchmark::min_time_of(
        [&]() { cgal_mesh = base_1m; },
        [&]() {
          PMP::isotropic_remeshing(faces(cgal_mesh), mult * mel_1m, cgal_mesh,
              CGAL::parameters::number_of_iterations(3));
          benchmark::do_not_optimize(cgal_mesh);
        },
        n_samples);

    out << mesh_1m.faces().size() << "," << mult << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
