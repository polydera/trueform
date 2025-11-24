/**
 * Boundary paths benchmark with TrueForm - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "boundary_paths-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_boundary_paths_tf_benchmark(const std::vector<std::string> &mesh_paths,
                                    int n_samples, std::ostream &out) {
  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto mesh = tf::read_stl<int>(path);
    auto polygons = mesh.polygons();

    std::size_t n_boundaries = 0;
    auto time = benchmark::min_time_of(
        [&]() {
          // Build topology and extract boundary paths
          auto boundary_paths = tf::make_boundary_paths(polygons);
          n_boundaries = boundary_paths.size();
          benchmark::do_not_optimize(boundary_paths);
        },
        n_samples);

    out << polygons.size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
