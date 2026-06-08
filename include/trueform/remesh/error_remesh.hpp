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
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
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
/// Returns the feature_handler (its face_labels are the post-remesh per-face
/// region labels for the preserve_regions case). No index maps -- the flip pass
/// makes original->final face maps meaningless; preserve_regions carries
/// per-face data instead.
///
/// @tparam Regions tf::none_t or tf::preserve_regions_t<...>.
template <typename Index, typename Real, std::size_t Dims, typename Regions>
auto error_remesh(tf::half_edges<Index> &he,
                  tf::points_buffer<Real, Dims> &points, Real error_rel,
                  int iterations, int optimize_iterations,
                  const tf::collapse_guard_config<Real> &cfg,
                  tf::rad<Real> feature_angle, Real lambda, int relaxation_iters,
                  Regions regions)
    -> feature_handler<Index, tf::remesh::region_label_t<Regions, Index>> {
  using Label = tf::remesh::region_label_t<Regions, Index>;
  auto bb = tf::aabb_from(points.points());
  Real diag = (bb.max - bb.min).length();
  if (diag <= Real(0))
    return feature_handler<Index, Label>{};
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

  feature_handler<Index, Label> features; // built once, reused throughout
  if constexpr (std::is_same_v<Regions, tf::none_t>) {
    if (feature_angle.value >= 0)
      features.init(he, points.points(), feature_angle);
  } else {
    if (feature_angle.value >= 0)
      features.init(he, points.points(), feature_angle, regions.face_regions);
    else
      features.init_regions(he, points.points(), regions.face_regions);
  }

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
    tf::improve_config<Real> icfg{optimize_iterations, relaxation_iters, lambda,
                                  cfg.check_normals,
                                  tf::flip_objective::min_angle};
    if (!features.empty())
      tf::remesh::improve_triangulation(he, points.points(), features.mask,
                                        old_pos, icfg);
    else
      tf::remesh::improve_triangulation(he, points.points(), tf::none, old_pos,
                                        icfg);
  }

  shift(Real(1));
  return features;
}

} // namespace tf::remesh
