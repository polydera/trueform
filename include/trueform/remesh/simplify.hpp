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

#include "../core/algorithm/parallel_copy.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/none.hpp"
#include "../core/points_buffer.hpp"
#include "./error_remesh.hpp"
#include "./preserve_regions.hpp"
#include "./simplify_config.hpp"

#include <utility>

namespace tf::remesh {

/// @ingroup remesh
/// @brief Error-budget simplify on an owned points buffer -- the shared
/// collapse + flip + relax loop (error_remesh) configured from simplify_config.
/// iterations = 1 is a single collapse + cleanup; iterations > 1 re-collapses
/// after each cleanup. Returns the feature_handler (its face_labels are the
/// post-op per-face region labels for the preserve_regions case). Regions is
/// tf::none_t or tf::preserve_regions_t<...>.
template <typename Index, typename Real, std::size_t Dims, typename Regions>
auto simplify(tf::half_edges<Index> &he, tf::points_buffer<Real, Dims> &points,
              const tf::simplify_config<Real> &config, Regions regions)
    -> feature_handler<Index, tf::remesh::region_label_t<Regions, Index>> {
  return tf::remesh::error_remesh(he, points, config.error_rel,
                                  config.iterations, config.optimize_iterations,
                                  config, config.feature_angle, config.lambda,
                                  config.relaxation_iters, regions);
}

} // namespace tf::remesh

namespace tf {

/// @ingroup remesh
/// @brief Simplify a mesh to a geometric error budget (quadric collapse), in
/// place on an owned points buffer. Flat regions collapse for ~0 error; curved
/// detail and features are kept. config.optimize_iterations rounds of min-angle
/// flip + relaxation clean up after each collapse; config.iterations > 1
/// re-collapses (an iterated remesh).
template <typename Index, typename Real, std::size_t Dims>
auto simplify(tf::half_edges<Index> &he, tf::points_buffer<Real, Dims> &points,
              const tf::simplify_config<Real> &config =
                  tf::simplify_config<Real>{}) -> void {
  tf::remesh::simplify(he, points, config, tf::none);
}

/// @ingroup remesh
/// @brief Region-preserving simplify (in place). Returns the post-op per-face
/// region labels.
template <typename Index, typename Real, std::size_t Dims, typename Range>
auto simplify(tf::half_edges<Index> &he, tf::points_buffer<Real, Dims> &points,
              const tf::simplify_config<Real> &config,
              tf::preserve_regions_t<Range> regions)
    -> tf::buffer<typename Range::value_type> {
  using Label = typename Range::value_type;
  // Empty range carries no labels: run the non-region path, return empty labels.
  if (regions.face_regions.size() == 0) {
    tf::simplify(he, points, config);
    return tf::buffer<Label>{};
  }
  auto features = tf::remesh::simplify(he, points, config, regions);
  return std::move(features.face_labels);
}

/// @ingroup remesh
/// @brief Overload taking const points; returns the simplified points buffer.
template <typename Index, typename PointsPolicy>
auto simplify(
    tf::half_edges<Index> &he, const tf::points<PointsPolicy> &points,
    const tf::simplify_config<tf::coordinate_type<PointsPolicy>> &config = {})
    -> tf::points_buffer<tf::coordinate_type<PointsPolicy>,
                         tf::coordinate_dims_v<PointsPolicy>> {
  using Real = tf::coordinate_type<PointsPolicy>;
  constexpr std::size_t Dims = tf::coordinate_dims_v<PointsPolicy>;
  tf::points_buffer<Real, Dims> buf;
  buf.allocate(points.size());
  tf::parallel_copy(points, buf.points());
  tf::simplify(he, buf, config);
  return buf;
}

/// @ingroup remesh
/// @brief Region-preserving overload taking const points; returns the new
/// points buffer paired with the post-op face labels.
template <typename Index, typename PointsPolicy, typename Range>
auto simplify(
    tf::half_edges<Index> &he, const tf::points<PointsPolicy> &points,
    const tf::simplify_config<tf::coordinate_type<PointsPolicy>> &config,
    tf::preserve_regions_t<Range> regions)
    -> std::pair<tf::points_buffer<tf::coordinate_type<PointsPolicy>,
                                   tf::coordinate_dims_v<PointsPolicy>>,
                 tf::buffer<typename Range::value_type>> {
  using Real = tf::coordinate_type<PointsPolicy>;
  constexpr std::size_t Dims = tf::coordinate_dims_v<PointsPolicy>;
  using Label = typename Range::value_type;
  // Empty range carries no labels: run the non-region path, return empty labels.
  if (regions.face_regions.size() == 0) {
    auto buf = tf::simplify(he, points, config);
    return {std::move(buf), tf::buffer<Label>{}};
  }
  tf::points_buffer<Real, Dims> buf;
  buf.allocate(points.size());
  tf::parallel_copy(points, buf.points());
  auto labels = tf::simplify(he, buf, config, regions);
  return {std::move(buf), std::move(labels)};
}

} // namespace tf
