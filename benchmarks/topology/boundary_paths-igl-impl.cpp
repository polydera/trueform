/**
 * Boundary paths benchmark with libigl - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "boundary_paths-igl.hpp"
#include "conversions.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <igl/boundary_loop.h>

namespace benchmark {

int run_boundary_paths_igl_benchmark(const std::vector<std::string> &mesh_paths,
                                     int n_samples, std::ostream &out) {
  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);

    // Convert to libigl format
    auto F = benchmark::igl::to_igl_faces(r_polygons.faces());

    auto time = benchmark::min_time_of(
        [&]() {
          std::vector<std::vector<int>> L;
          ::igl::boundary_loop(F, L);
          benchmark::do_not_optimize(L);
        },
        n_samples);

    out << r_polygons.faces().size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
