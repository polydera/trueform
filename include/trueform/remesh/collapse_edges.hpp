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

#include "./collapse/collapse_to_exhaustion.hpp"
#include "./collapse/collapse_to_exhaustion_parallel.hpp"
#include "./collapse/collapse_to_target.hpp"
#include "./collapse/collapse_to_target_parallel.hpp"
#include "./collapse_handler.hpp"

namespace tf {

/// @ingroup remesh
/// @brief Collapse edges to exhaustion using a collapse_handler.
template <typename Index, typename PointsPolicy, typename Real,
          typename ScoreFn, typename AllowedFn>
auto collapse_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    tf::collapse_handler<Real, ScoreFn, AllowedFn> &handler) -> Index {
  handler.init(he, points);
  if (handler.parallel())
    return tf::remesh::collapse_to_exhaustion_parallel<1024>(he, points,
                                                             handler);
  return tf::remesh::collapse_to_exhaustion(he, points, handler);
}

/// @ingroup remesh
/// @brief Collapse edges to a target face count using a collapse_handler.
template <typename Index, typename PointsPolicy, typename Real,
          typename ScoreFn, typename AllowedFn>
auto collapse_edges(
    tf::half_edges<Index> &he, tf::points<PointsPolicy> &points,
    tf::collapse_handler<Real, ScoreFn, AllowedFn> &handler,
    Index target_faces) -> Index {
  handler.init(he, points);
  Index current_faces = he.number_of_faces();
  if (handler.parallel())
    return tf::remesh::collapse_to_target_parallel<1024>(
        he, points, current_faces, target_faces, handler);
  return tf::remesh::collapse_to_target(he, points, current_faces,
                                        target_faces, handler);
}

} // namespace tf
