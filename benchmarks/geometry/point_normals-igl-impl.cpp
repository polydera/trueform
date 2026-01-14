/**
 * Point normals benchmark with libigl - Implementation
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "point_normals-igl.hpp"
#include "conversions.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

#include <igl/per_vertex_normals.h>

namespace benchmark {

int run_point_normals_igl_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {
  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto mesh = tf::read_stl<int>(path);

    // Convert to libigl format
    auto V = benchmark::igl::to_igl_vertices(mesh.points());
    auto F = benchmark::igl::to_igl_faces(mesh.faces());

    Eigen::MatrixXd N;

    auto time = benchmark::min_time_of(
        [&]() {
          ::igl::per_vertex_normals(V, F, N);
          benchmark::do_not_optimize(N);
        },
        n_samples);

    out << mesh.faces().size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
