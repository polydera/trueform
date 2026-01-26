/**
 * Connected components benchmark with TrueForm - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "connected_components-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_connected_components_tf_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto mesh = tf::read_stl<int>(path);
    auto polygons = mesh.polygons();

    short n_components = 0;
    auto time = benchmark::min_time_of(
        [&]() {
          // no precompute of topology
          auto [labels, n_components] =
              tf::make_manifold_edge_connected_component_labels(polygons);
          benchmark::do_not_optimize(labels);
        },
        n_samples);

    out << polygons.size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
