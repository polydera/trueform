/**
 * Isotropic remeshing benchmark with TrueForm - Implementation
 *
 * Remeshes each mesh with target_length = 2× mean_edge_length,
 * 3 iterations. Half-edges are precomputed to match CGAL's
 * Surface_mesh which includes built-in half-edge connectivity.
 * Then sweeps multipliers 0.5×–4× on the 1M mesh.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "isotropic_remeshing-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_isotropic_remeshing_tf_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,multiplier,time_ms\n";

  // All mesh sizes at 2× mel
  for (const auto &path : mesh_paths) {
    auto mesh = tf::read_stl<int>(path);
    auto polys = mesh.polygons();
    tf::half_edges<int> he(polys);
    auto tagged = polys | tf::tag(he);
    float mel = tf::mean_edge_length(polys);

    tf::remesh_config<float> config;
    config.target_length = 2.0f * mel;

    auto time = benchmark::min_time_of(
        [&]() {
          auto [result, he_out] = tf::isotropic_remeshed(tagged, config);
          benchmark::do_not_optimize(result);
        },
        n_samples);

    out << polys.size() << ",2," << time << "\n";
  }

  // 1M mesh at multipliers 0.5–4×
  auto mesh_1m = tf::read_stl<int>(mesh_paths.back());
  auto polys_1m = mesh_1m.polygons();
  tf::half_edges<int> he_1m(polys_1m);
  auto tagged_1m = polys_1m | tf::tag(he_1m);
  float mel_1m = tf::mean_edge_length(polys_1m);

  float multipliers[] = {1.0f, 2.0f, 3.0f, 4.0f};
  for (float mult : multipliers) {
    tf::remesh_config<float> config;
    config.target_length = mult * mel_1m;

    auto time = benchmark::min_time_of(
        [&]() {
          auto [result, he_out] = tf::isotropic_remeshed(tagged_1m, config);
          benchmark::do_not_optimize(result);
        },
        n_samples);

    out << polys_1m.size() << "," << mult << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
