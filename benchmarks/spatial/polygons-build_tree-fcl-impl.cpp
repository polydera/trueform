/**
 * Polygons tree building FCL - Implementation
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "conversions.hpp"
#include "polygons-build_tree-fcl.hpp"
#include "timing.hpp"
#include <memory>
#include <trueform/trueform.hpp>

using namespace benchmark::fcl;

namespace benchmark {

int run_polygons_build_tree_fcl_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {

  out << "polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);

    // Convert to FCL geometry
    auto [fcl_vertices, fcl_triangles] =
        benchmark::fcl::to_fcl_geometry(r_polygons);

    if (fcl_vertices.empty() || fcl_triangles.empty()) {
      continue; // skipped due to empty or non-triangular mesh
    }

    auto time = benchmark::min_time_of(
        [&]() {
          auto model = std::make_shared<Model>();
          model->beginModel(static_cast<int>(fcl_triangles.size()),
                            static_cast<int>(fcl_vertices.size()));
          model->addSubModel(fcl_vertices, fcl_triangles);
          model->endModel();
          benchmark::do_not_optimize(model);
        },
        n_samples);

    out << fcl_triangles.size() << "," << time << "\n";
  }

  return 0;
}

} // namespace benchmark
