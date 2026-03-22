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

#include "./compare_faces.hpp"

namespace tf {

/// @ingroup topology_analysis
/// @brief Checks if two faces are equal regardless of winding order.
///
/// Faces are cyclic sequences of vertex indices. Two faces are considered
/// equal if they have identical vertices in either the same or reversed
/// cyclic order. For example, {0,1,2} equals both {1,2,0} and {0,2,1}.
///
/// @see tf::are_oriented_faces_equal() for orientation-sensitive comparison.
/// @see tf::compare_faces() for full three-way comparison.
template <typename Face1, typename Face2>
auto are_faces_equal(const Face1 &face, const Face2 &neighbor) -> bool {
  return tf::compare_faces(face, neighbor) != 0;
}

} // namespace tf
