/**
 * Decimation benchmark with TrueForm - Implementation
 *
 * Decimates each mesh to 50% face count using quadric error metrics,
 * then sweeps ratios 0.9-0.1 on the 1M mesh.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "decimation-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_decimation_tf_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,ratio,time_ms\n";

  tf::decimate_config<float> config;
  config.max_aspect_ratio = -1;

  // All mesh sizes at ratio 0.5
  for (const auto &path : mesh_paths) {
    auto mesh = tf::read_stl<int>(path);
    auto polys = mesh.polygons();
    tf::half_edges<int> he(polys);
    auto tagged = polys | tf::tag(he);

    auto time = benchmark::min_time_of(
        [&]() {
          auto [result, he_out] = tf::decimated(tagged, 0.5f, config);
          benchmark::do_not_optimize(result);
        },
        n_samples);

    out << polys.size() << ",0.5," << time << "\n";
  }

  // 1M mesh at ratios 0.9 down to 0.1
  auto mesh_1m = tf::read_stl<int>(mesh_paths.back());
  auto polys_1m = mesh_1m.polygons();
  tf::half_edges<int> he_1m(polys_1m);
  auto tagged_1m = polys_1m | tf::tag(he_1m);

  for (int r = 9; r >= 1; --r) {
    float ratio = r * 0.1f;

    auto time = benchmark::min_time_of(
        [&]() {
          auto [result, he_out] = tf::decimated(tagged_1m, ratio, config);
          benchmark::do_not_optimize(result);
        },
        n_samples);

    out << polys_1m.size() << "," << ratio << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
