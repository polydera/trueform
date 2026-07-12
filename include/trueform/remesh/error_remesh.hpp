/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#pragma once

#include "../core/aabb_from.hpp"
#include "../core/algorithm/compose_index_maps.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/algorithm/parallel_iota.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/index_map.hpp"
#include "../core/none.hpp"
#include "../core/points_buffer.hpp"
#include "../core/views/sequence_range.hpp"
#include "../reindex/points.hpp"
#include "./collapse_checker.hpp"
#include "./collapse_edges.hpp"
#include "./collapse_guard_config.hpp"
#include "./collapse_handler.hpp"
#include "./feature_handler.hpp"
#include "./improve_config.hpp"
#include "./improve_triangulation.hpp"
#include "./preserve_regions.hpp"
#include "./regions/region_label.hpp"

#include <type_traits>
#include <utility>

namespace tf::remesh {

/// @brief One quadric error-budget collapse pass on a points VIEW, reusing an
/// already-built feature_handler. Assumes points are pre-centered (caller owns
/// the bbox shift). inv_eps = 1 / (error_rel * diagonal): a collapse is in
/// budget when its quadric error * inv_eps <= 1. Returns n_collapsed.
template <typename Index, typename PointsPolicy, typename Real, typename Label>
auto error_collapse(tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
                    const tf::collapse_guard_config<Real> &cfg, Real inv_eps,
                    feature_handler<Index, Label> &features) -> Index {
  auto score = [inv_eps](const auto &he, const auto &points, auto heh,
                         const auto &handler) -> Real {
    Real e = tf::remesh::collapse_error_quadric<Real>(
        handler._quadrics, points, he, heh, handler._config.stabilizer);
    return e * inv_eps;
  };
  auto checker = tf::make_collapse_checker<Real>(cfg.min_quality, tf::none,
                                                 cfg.check_normals);
  auto handler = tf::make_collapse_handler<Real>(score, checker,
                                                 features.as_view(), cfg);
  return tf::collapse_edges(he, points, handler);
}

/// @ingroup remesh
/// @brief Error-budget remesh, in place on an owned points buffer. The shared
/// core of tf::remesh and (at iterations = 1) tf::simplify.
///
/// Builds the feature_handler ONCE and maintains it (compact + recompute), then
/// loops: collapse to the error budget, compact/reindex, recompute the feature
/// mask, run optimize_iterations rounds of min-angle flip + relaxation. The
/// bbox-centering shift and the error diagonal are hoisted out of the loop, so
/// the budget is anchored to the original mesh size for every iteration.
///
/// Returns @ref remesh_result: the feature_handler (its face_labels are
/// the post-remesh per-face region labels for the preserve_regions case) and,
/// when ReturnMap is requested, the original->final vertex index map composed
/// across every pass.
///
/// @tparam Regions tf::none_t or tf::preserve_regions_t<...>.
/// @tparam Protection tf::none_t or tf::protect_vertices_t<...>.
/// @tparam ReturnMap tf::none_t or tf::return_index_map_t.
template <typename Index, typename Real, std::size_t Dims, typename Regions,
          typename Protection, typename ReturnMap>
auto error_remesh(tf::half_edges<Index> &he,
                  tf::points_buffer<Real, Dims> &points, Real error_rel,
                  int iterations, int optimize_iterations,
                  const tf::collapse_guard_config<Real> &cfg,
                  tf::rad<Real> feature_angle, Real lambda, int relaxation_iters,
                  Regions regions, Protection protection, ReturnMap)
    -> remesh_result<Index, tf::remesh::region_label_t<Regions, Index>> {
  using Label = tf::remesh::region_label_t<Regions, Index>;
  constexpr bool want_map = !std::is_same_v<ReturnMap, tf::none_t>;
  auto n0 = Index(points.size());
  auto bb = tf::aabb_from(points.points());
  Real diag = (bb.max - bb.min).length();
  if (diag <= Real(0))
    return remesh_result<Index, Label>{};
  Real inv_eps = Real(1) / (error_rel * diag);

  // Center on bbox-min once (quadric numerical stability at large coordinates);
  // shift back once at the end. Coarsening keeps the bbox put, so one centering
  // holds for every iteration.
  auto origin = bb.min;
  auto shift = [&points, origin](Real sign) {
    tf::parallel_for_each(tf::make_sequence_range(Index(points.size())),
                          [&](Index i) {
                            for (std::size_t d = 0; d < Dims; ++d)
                              points.points()[i][d] += sign * origin[d];
                          });
  };
  shift(Real(-1));

  // Built once, reused throughout (protection re-applied by every recompute).
  auto features = tf::remesh::build_feature_handler(
      he, points.points(), feature_angle, regions, protection);

  // Composed original->current vertex map, folded one pass at a time.
  tf::index_map_buffer<Index> vmap;
  bool vmap_seeded = false;

  tf::points_buffer<double, Dims> old_pos;
  for (int i = 0; i < iterations; ++i) {
    {
      auto pv = points.points();
      tf::remesh::error_collapse(he, pv, cfg, inv_eps, features);
    }
    auto [face_im, vert_im, edge_im] = he.compact();
    points = tf::reindexed(points.points(), vert_im);
    if (!features.empty()) {
      features.compact(face_im, edge_im, vert_im);
      features.recompute(he, points.points(), feature_angle);
    }
    if constexpr (want_map) {
      // vert_im maps this pass's old->new ids; fold it into the running map.
      if (!vmap_seeded) {
        vmap = std::move(vert_im);
        vmap_seeded = true;
      } else
        vmap = tf::compose_index_maps(vmap, vert_im);
    }
    tf::improve_config<Real> icfg{optimize_iterations, relaxation_iters, lambda,
                                  cfg.check_normals,
                                  tf::flip_objective::min_angle,
                                  error_rel * diag};
    if (!features.empty())
      tf::remesh::improve_triangulation(he, points.points(), features.mask,
                                        old_pos, icfg);
    else
      tf::remesh::improve_triangulation(he, points.points(), tf::none, old_pos,
                                        icfg);
  }

  shift(Real(1));

  if constexpr (want_map) {
    // No pass ran (iterations == 0): the vertex set is unchanged, so the map is
    // the identity over the original vertices.
    if (!vmap_seeded) {
      vmap.f().allocate(n0);
      vmap.kept_ids().allocate(n0);
      tf::parallel_iota(tf::make_range(vmap.f()), Index(0));
      tf::parallel_iota(tf::make_range(vmap.kept_ids()), Index(0));
    }
  }
  return remesh_result<Index, Label>{std::move(features), std::move(vmap)};
}

} // namespace tf::remesh
