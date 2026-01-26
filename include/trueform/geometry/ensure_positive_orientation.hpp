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
#include "../core/polygons.hpp"
#include "../core/signed_volume.hpp"
#include "../topology/orient_faces_consistently.hpp"
#include "../topology/reverse_winding.hpp"

namespace tf {

/// @ingroup geometry_processing
/// @brief Ensures mesh faces are oriented with outward-pointing normals.
///
/// For closed meshes, orients all faces consistently and ensures the signed
/// volume is positive (normals point outward).
///
/// @tparam Policy The policy type of the polygons.
/// @param polygons The mesh to orient (modified in place).
/// @param is_consistent If true, skips the orient_faces_consistently step.
template <typename Policy>
auto ensure_positive_orientation(tf::polygons<Policy> &polygons,
                                 bool is_consistent = false) -> void {
  if (!is_consistent)
    tf::orient_faces_consistently(polygons);
  auto stripped = tf::make_polygons(polygons.faces(), polygons.points());
  if (tf::signed_volume(stripped) < 0)
    tf::reverse_winding(polygons.faces());
}

/// @overload
template <typename Policy>
auto ensure_positive_orientation(tf::polygons<Policy> &&polygons,
                                 bool is_consistent = false) -> void {
  ensure_positive_orientation(polygons, is_consistent);
}
} // namespace tf
