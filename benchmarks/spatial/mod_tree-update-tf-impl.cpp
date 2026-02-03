/**
 * mod_tree update benchmark with TrueForm - Implementation
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include "mod_tree-update-tf.hpp"
#include "timing.hpp"
#include <trueform/trueform.hpp>

namespace benchmark {

int run_mod_tree_update_tf_benchmark(
    const std::vector<std::string> &mesh_paths,
    int n_samples,
    std::ostream &out) {

  out << "polygons,dirty_pct,update_time_ms,full_build_ms,update_pct\n";

  for (const auto &path : mesh_paths) {
    auto mesh = tf::read_stl<int>(path);
    auto polygons = mesh.polygons();
    const auto n_polys = polygons.faces().size();
    const auto n_verts = polygons.points().size();
    if(n_polys < 100000) continue;

    auto vlink = tf::make_vertex_link(polygons);
    auto fm = tf::make_face_membership(polygons);

    tf::aabb_mod_tree<int, float, 3> tree;
    std::vector<int> dirty_ids;
    std::vector<char> dirty_mask(n_polys, 0);
    tf::topology::neighborhood_applier<int> applier;

    const auto full_build_ms = benchmark::min_time_of(
        [&]() { tree = tf::aabb_mod_tree<int, float, 3>{}; },
        [&]() {
          tree.build(polygons, tf::config_tree(4, 4));
          benchmark::do_not_optimize(tree);
        },
        n_samples);

    for (int pct = 1; pct <= 40; ++pct) {
      const double dirty_pct = pct;
      const std::size_t n_dirty =
          static_cast<std::size_t>((n_polys * dirty_pct) / 100.0);

      if (n_dirty == 0)
        continue;

      auto update_time_ms = benchmark::min_time_of(
          [&]() {
            tree.build(polygons, tf::config_tree(4, 4));

            for (int id : dirty_ids)
              dirty_mask[id] = 0;
            dirty_ids.clear();

            int seed = tf::random<int>(0, static_cast<int>(n_verts) - 1);

            applier(
                vlink, seed,
                [&](int, int) {
                  return dirty_ids.size() < n_dirty ? 0.0f : 1.0f;
                },
                0.5f,
                [&](int vid) {
                  for (auto poly_id : fm[vid]) {
                    if (dirty_ids.size() >= n_dirty)
                      return;
                    if (!dirty_mask[poly_id]) {
                      dirty_mask[poly_id] = 1;
                      dirty_ids.push_back(poly_id);
                    }
                  }
                },
                true);
          },
          [&]() {
            tree.update(
                polygons, dirty_ids,
                [&](int id) { return !dirty_mask[id]; },
                tf::config_tree(4, 4));
            benchmark::do_not_optimize(tree);
          },
          n_samples);

      const double update_pct = (update_time_ms / full_build_ms) * 100.0;

      out << n_polys << "," << dirty_pct << "," << update_time_ms << ","
          << full_build_ms << "," << update_pct << "\n";
    }
  }

  return 0;
}

} // namespace benchmark
