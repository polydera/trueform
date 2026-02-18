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

#include "../core/coordinate_type.hpp"
#include "./collapse/collapse_to_target.hpp"
#include "./collapse/collapse_to_target_parallel.hpp"
#include "./collapse_policy.hpp"
#include "./decimate_config.hpp"

namespace tf {

/// @ingroup remesh
/// @brief Decimate a mesh to a target face count using quadric error metrics.
///
/// Dispatches to sequential or parallel implementation based on
/// `config.parallel` (default true).
///
/// @param he The half-edge structure (modified in place).
/// @param points The vertex positions (modified in place).
/// @param target_proportion The proportion of faces to keep (0.0 to 1.0).
/// @param config Decimation configuration.
/// @return The number of edges collapsed.
template <typename Index, typename PointsPolicy>
auto decimate(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    tf::coordinate_type<PointsPolicy> target_proportion,
    const tf::decimate_config<tf::coordinate_type<PointsPolicy>> &config =
        tf::decimate_config<tf::coordinate_type<PointsPolicy>>{})
    -> Index {
  using Real = tf::coordinate_type<PointsPolicy>;
  Index current_faces = he.number_of_faces();
  Index target_faces = Index(current_faces * target_proportion);
  tf::collapse_policy<Real> policy{config.max_aspect_ratio,
                                   config.preserve_boundary, config.stabilizer};
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
auto decimate(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &&points,
    tf::coordinate_type<PointsPolicy> target_proportion,
    const tf::decimate_config<tf::coordinate_type<PointsPolicy>> &config)
    -> Index {
  return tf::decimate(he, points, target_proportion, config);
}

} // namespace tf
