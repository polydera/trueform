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
#include "../arrangement/arrangement_config.hpp"
#include "../arrangement/make_arrangement_graph.hpp"
#include "../core/none.hpp"
#include "../core/polygons.hpp"
#include "./intersect_config.hpp"
#include "./make_intersection_curves.hpp"

namespace tf {

/// @ingroup intersect_curves
/// @brief Extract curves where a mesh intersects itself.
///
/// Finds all locations where a mesh's faces intersect each other
/// (excluding adjacent faces) and returns the result as connected curves.
///
/// @tparam Policy The policy type for the mesh.
/// @param _polygons The input @ref tf::polygons (or tagged form).
/// @param config The intersection configuration (mode and tolerance).
/// @return A @ref tf::curves_buffer containing connected self-intersection
/// curves.
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy>
auto make_self_intersection_curves(
    const tf::polygons<Policy> &_polygons,
    tf::intersect_config config = {tf::intersect_mode::primitives |
                                   tf::intersect_mode::resolve_contours}) {
  return intersect::curves_worker<OutputCoordinateType>(
      tf::make_arrangement_graph<Int>(_polygons,
                                       tf::arrangement_config{config}));
}

} // namespace tf
