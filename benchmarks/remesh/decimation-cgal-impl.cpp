/**
 * Decimation benchmark with CGAL - Implementation
 *
 * Decimates each mesh to 50% face count using
 * CGAL::Surface_mesh_simplification::edge_collapse,
 * then sweeps ratios 0.9-0.1 on the 1M mesh.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "decimation-cgal.hpp"
#include "conversions.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_count_ratio_stop_predicate.h>

namespace benchmark {

namespace SMS = CGAL::Surface_mesh_simplification;

int run_decimation_cgal_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,ratio,time_ms\n";

  // All mesh sizes at ratio 0.5
  for (const auto &path : mesh_paths) {
    auto mesh = tf::read_stl<int>(path);
    auto base_mesh = benchmark::cgal::to_cgal_mesh(mesh);

    cgal::Surface_mesh cgal_mesh;

    auto time = benchmark::min_time_of(
        [&]() { cgal_mesh = base_mesh; },
        [&]() {
          SMS::edge_collapse(cgal_mesh,
              SMS::Edge_count_ratio_stop_predicate<cgal::Surface_mesh>(0.5));
          benchmark::do_not_optimize(cgal_mesh);
        },
        n_samples);

    out << mesh.faces().size() << ",0.5," << time << "\n";
  }

  // 1M mesh at ratios 0.9 down to 0.1
  auto mesh_1m = tf::read_stl<int>(mesh_paths.back());
  auto base_1m = benchmark::cgal::to_cgal_mesh(mesh_1m);

  for (int r = 9; r >= 1; --r) {
    float ratio = r * 0.1f;
    cgal::Surface_mesh cgal_mesh;

    auto time = benchmark::min_time_of(
        [&]() { cgal_mesh = base_1m; },
        [&]() {
          SMS::edge_collapse(cgal_mesh,
              SMS::Edge_count_ratio_stop_predicate<cgal::Surface_mesh>(ratio));
          benchmark::do_not_optimize(cgal_mesh);
        },
        n_samples);

    out << mesh_1m.faces().size() << "," << ratio << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
