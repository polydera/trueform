/**
 * STL reading benchmark with libigl - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "read_stl-igl.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <igl/readSTL.h>
#include <Eigen/Core>
#include <fstream>

namespace benchmark {

int run_read_stl_igl_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    // First read to get polygon count
    auto mesh_for_count = tf::read_stl<int>(path);
    const auto n_polygons = mesh_for_count.faces().size();

    Eigen::MatrixXd V, N;
    Eigen::MatrixXi F;

    auto time = benchmark::min_time_of(
        [&]() {
          std::ifstream input(path, std::ios::binary);
          ::igl::readSTL(input, V, F, N);
          benchmark::do_not_optimize(V);
          benchmark::do_not_optimize(F);
        },
        n_samples);

    out << n_polygons << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
