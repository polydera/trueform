/**
 * Polygons tree building Coal - Implementation
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "conversions.hpp"
#include "polygons-build_tree-coal.hpp"
#include "timing.hpp"
#include <memory>
#include <trueform/trueform.hpp>

using namespace benchmark::coal;

namespace benchmark {

int run_polygons_build_tree_coal_benchmark(
    const std::vector<std::string> &mesh_paths, int n_samples,
    std::ostream &out) {

  out << "bv,polygons,time_ms\n";

  for (const auto &path : mesh_paths) {
    auto r_polygons = tf::read_stl<int>(path);

    auto [coal_vertices, coal_triangles] =
        benchmark::coal::to_coal_geometry(r_polygons);

    if (coal_vertices.empty() || coal_triangles.empty()) {
      continue;
    }

    auto n_tris = static_cast<int>(coal_triangles.size());
    auto n_verts = static_cast<int>(coal_vertices.size());

    auto time_aabb = benchmark::min_time_of(
        [&]() {
          auto model = std::make_shared<Model_AABB>();
          model->beginModel(n_tris, n_verts);
          model->addSubModel(coal_vertices, coal_triangles);
          model->endModel();
          benchmark::do_not_optimize(model);
        },
        n_samples);
    out << "AABB," << coal_triangles.size() << "," << time_aabb << "\n";

    auto time_obb = benchmark::min_time_of(
        [&]() {
          auto model = std::make_shared<Model_OBB>();
          model->beginModel(n_tris, n_verts);
          model->addSubModel(coal_vertices, coal_triangles);
          model->endModel();
          benchmark::do_not_optimize(model);
        },
        n_samples);
    out << "OBB," << coal_triangles.size() << "," << time_obb << "\n";

    auto time_obbrss = benchmark::min_time_of(
        [&]() {
          auto model = std::make_shared<Model_OBBRSS>();
          model->beginModel(n_tris, n_verts);
          model->addSubModel(coal_vertices, coal_triangles);
          model->endModel();
          benchmark::do_not_optimize(model);
        },
        n_samples);
    out << "OBBRSS," << coal_triangles.size() << "," << time_obbrss << "\n";
  }

  return 0;
}

} // namespace benchmark
