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

  out << "bv,polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);

    auto [fcl_vertices, fcl_triangles] =
        benchmark::fcl::to_fcl_geometry(r_polygons);

    if (fcl_vertices.empty() || fcl_triangles.empty()) {
      continue;
    }

    auto n_tris = static_cast<int>(fcl_triangles.size());
    auto n_verts = static_cast<int>(fcl_vertices.size());

    auto time_aabb = benchmark::min_time_of(
        [&]() {
          auto model = std::make_shared<Model_AABB>();
          model->beginModel(n_tris, n_verts);
          model->addSubModel(fcl_vertices, fcl_triangles);
          model->endModel();
          benchmark::do_not_optimize(model);
        },
        n_samples);
    out << "AABB," << fcl_triangles.size() << "," << time_aabb << "\n";

    auto time_obb = benchmark::min_time_of(
        [&]() {
          auto model = std::make_shared<Model_OBB>();
          model->beginModel(n_tris, n_verts);
          model->addSubModel(fcl_vertices, fcl_triangles);
          model->endModel();
          benchmark::do_not_optimize(model);
        },
        n_samples);
    out << "OBB," << fcl_triangles.size() << "," << time_obb << "\n";

    auto time_obbrss = benchmark::min_time_of(
        [&]() {
          auto model = std::make_shared<Model_OBBRSS>();
          model->beginModel(n_tris, n_verts);
          model->addSubModel(fcl_vertices, fcl_triangles);
          model->endModel();
          benchmark::do_not_optimize(model);
        },
        n_samples);
    out << "OBBRSS," << fcl_triangles.size() << "," << time_obbrss << "\n";
  }

  return 0;
}

} // namespace benchmark
