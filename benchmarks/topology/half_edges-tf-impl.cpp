/**
 * Half-edge construction benchmark with TrueForm - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "half_edges-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_half_edges_tf_benchmark(const std::vector<std::string> &mesh_paths,
                                int n_samples, std::ostream &out) {
  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto polygons = tf::read_stl<int>(path);

    auto time = benchmark::min_time_of(
        [&]() {
          tf::half_edges<int> he;
          he.build(polygons.polygons());
          benchmark::do_not_optimize(he);
        },
        n_samples);

    out << polygons.faces().size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
