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
#include "./collapse_checker.hpp"
#include "./collapse_edges.hpp"
#include "./collapse_handler.hpp"
#include "./decimate_config.hpp"

namespace tf {

/// @ingroup remesh
/// @brief Decimate a mesh to a target face count using quadric error metrics.
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
  Index target_faces = Index(he.number_of_faces() * target_proportion);

  auto score = [](const auto &he, const auto &points, auto heh,
                  const auto &handler) -> Real {
    return tf::remesh::collapse_error_quadric<Real>(
        handler._quadrics, points, he, heh, handler._config.stabilizer);
  };

  auto checker = tf::make_collapse_checker<Real>(config.max_aspect_ratio);
  auto handler = tf::make_collapse_handler<Real>(score, checker, config);
  return tf::collapse_edges(he, points, handler, target_faces);
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
