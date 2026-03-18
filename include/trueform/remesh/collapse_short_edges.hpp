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

#include <limits>

#include "../core/coordinate_type.hpp"
#include "./collapse/collapse_to_exhaustion.hpp"
#include "./collapse/collapse_to_exhaustion_parallel.hpp"
#include "./collapse/collapse_to_target.hpp"
#include "./collapse/collapse_to_target_parallel.hpp"
#include "./length_collapse_config.hpp"
#include "./length_collapse_policy.hpp"

namespace tf {

// ---------------------------------------------------------------------------
// Exhaustion mode — collapse all edges shorter than min_len
// ---------------------------------------------------------------------------

/// @ingroup remesh
/// @brief Collapse edges shorter than min_len.
///
/// Dispatches to sequential or parallel implementation based on
/// `config.parallel` (default true).
template <typename Index, typename PointsPolicy>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    tf::coordinate_type<PointsPolicy> min_len,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>>
        &config =
            tf::length_collapse_config<tf::coordinate_type<PointsPolicy>>{})
    -> Index {
  using Real = tf::coordinate_type<PointsPolicy>;
  tf::length_collapse_policy<Real> policy(min_len, config.max_length,
                                          config.preserve_boundary,
                                          config.use_quadric,
                                          config.max_aspect_ratio,
                                          config.stabilizer,
                                          config.feature_angle,
                                          config.feature_weight);
  if (config.parallel) {
    return tf::remesh::collapse_to_exhaustion_parallel<1024>(he, points,
                                                             policy);
  } else {
    policy.init(he, points);
    return tf::remesh::collapse_to_exhaustion(he, points, policy);
  }
}

template <typename Index, typename PointsPolicy>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    tf::coordinate_type<PointsPolicy> min_len,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>>
        &config) -> Index {
  return tf::collapse_short_edges(he, points, min_len, config);
}

// ---------------------------------------------------------------------------
// Target mode — collapse shortest edges until target face count is reached
// ---------------------------------------------------------------------------

/// @ingroup remesh
/// @brief Collapse short edges to reach a target face count.
///
/// Dispatches to sequential or parallel implementation based on
/// `config.parallel` (default true).
template <typename Index, typename PointsPolicy>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    Index target_faces,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>>
        &config =
            tf::length_collapse_config<tf::coordinate_type<PointsPolicy>>{})
    -> Index {
  using Real = tf::coordinate_type<PointsPolicy>;
  Index current_faces = he.number_of_faces();
  tf::length_collapse_policy<Real> policy(Real(0), config.max_length,
                                          config.preserve_boundary,
                                          config.use_quadric,
                                          config.max_aspect_ratio,
                                          config.stabilizer,
                                          config.feature_angle,
                                          config.feature_weight);
  policy.init(he, points);
  if (config.parallel) {
    return tf::remesh::collapse_to_target_parallel<1024>(
        he, points, current_faces, target_faces, policy);
  } else {
    return tf::remesh::collapse_to_target(he, points, current_faces,
                                          target_faces, policy);
  }
}

template <typename Index, typename PointsPolicy>
auto collapse_short_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    Index target_faces,
    const tf::length_collapse_config<tf::coordinate_type<PointsPolicy>>
        &config) -> Index {
  return tf::collapse_short_edges(he, points, target_faces, config);
}

} // namespace tf
