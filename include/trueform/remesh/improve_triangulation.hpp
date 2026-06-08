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

#include <type_traits>

#include "../core/buffer.hpp"
#include "../core/constants.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/none.hpp"
#include "../core/points.hpp"
#include "../core/points_buffer.hpp"
#include "../core/angle.hpp"
#include "../topology/half_edges.hpp"
#include "./feature_mask.hpp"
#include "./flip_min_angle.hpp"
#include "./improve_config.hpp"
#include "./optimize_valence.hpp"
#include "./tangential_relaxation.hpp"
#include "./valence_deviation.hpp"

namespace tf::remesh {

/// @ingroup remesh
/// @brief Improve a triangulation at fixed vertex/face count: rounds of an edge
/// flip pass followed by tangential relaxation.
///
/// `mask` is a feature_mask (features/curves are preserved) or tf::none.
/// `old_pos` is the relaxation workspace; pass one buffer to reuse it across
/// calls (e.g. from the isotropic remesh loop).
template <typename Index, typename PointsPolicy, typename MaskOrNone>
auto improve_triangulation(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> points,
    const MaskOrNone &mask,
    tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> &old_pos,
    const tf::improve_config<tf::coordinate_type<PointsPolicy>> &config) -> void {
  using Real = tf::coordinate_type<PointsPolicy>;
  constexpr bool HasMask = !std::is_same_v<MaskOrNone, tf::none_t>;
  const tf::rad<Real> angle_floor(Real(20) * tf::pi<Real> / Real(180));

  // Valence deviations are only needed for the valence objective; the flip pass
  // updates them in place, so they stay valid across rounds (relaxation does not
  // change topology).
  tf::buffer<std::int8_t> deviation;
  if (config.flip == tf::flip_objective::valence)
    deviation = tf::remesh::compute_valence_deviations(he);

  for (int i = 0; i < config.iterations; ++i) {
    if (config.flip == tf::flip_objective::valence) {
      if constexpr (HasMask) {
        if (config.check_normals)
          tf::remesh::optimize_valence(he, points, deviation, mask, 1);
        else
          tf::remesh::optimize_valence(he, deviation, mask, 1);
      } else {
        if (config.check_normals)
          tf::remesh::optimize_valence(he, points, deviation, 1);
        else
          tf::remesh::optimize_valence(he, deviation, 1);
      }
    } else {
      tf::remesh::flip_min_angle(he, points, mask, 1, angle_floor);
    }

    if constexpr (HasMask)
      tf::remesh::tangential_relaxation(he, points, old_pos, mask,
                                        config.relaxation_iters, config.lambda);
    else
      tf::remesh::tangential_relaxation(he, points, old_pos,
                                        config.relaxation_iters, config.lambda);
  }
}

} // namespace tf::remesh

namespace tf {

/// @ingroup remesh
/// @brief Improve a triangulation (flips + relaxation), feature-aware, with a
/// caller-owned relaxation workspace.
template <typename Index, typename PointsPolicy>
auto improve_triangulation(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> points,
    const tf::remesh::feature_mask &mask,
    tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> &old_pos,
    const tf::improve_config<tf::coordinate_type<PointsPolicy>> &config = {})
    -> void {
  tf::remesh::improve_triangulation(he, points, mask, old_pos, config);
}

/// @ingroup remesh
/// @brief Improve a triangulation, feature-aware (allocates the workspace).
template <typename Index, typename PointsPolicy>
auto improve_triangulation(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> points,
    const tf::remesh::feature_mask &mask,
    const tf::improve_config<tf::coordinate_type<PointsPolicy>> &config = {})
    -> void {
  tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> old_pos;
  tf::remesh::improve_triangulation(he, points, mask, old_pos, config);
}

/// @ingroup remesh
/// @brief Improve a triangulation with a caller-owned relaxation workspace.
template <typename Index, typename PointsPolicy>
auto improve_triangulation(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> points,
    tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> &old_pos,
    const tf::improve_config<tf::coordinate_type<PointsPolicy>> &config = {})
    -> void {
  tf::remesh::improve_triangulation(he, points, tf::none, old_pos, config);
}

/// @ingroup remesh
/// @brief Improve a triangulation (allocates the workspace).
template <typename Index, typename PointsPolicy>
auto improve_triangulation(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> points,
    const tf::improve_config<tf::coordinate_type<PointsPolicy>> &config = {})
    -> void {
  tf::points_buffer<double, tf::coordinate_dims_v<PointsPolicy>> old_pos;
  tf::remesh::improve_triangulation(he, points, tf::none, old_pos, config);
}

} // namespace tf
