/*
 * Copyright (c) 2026 XLAB
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
#include "../core/algorithm/parallel_contains.hpp"
#include "../core/checked.hpp"
#include "../core/faces.hpp"
#include "../core/polygons.hpp"
#include "../core/views/sequence_range.hpp"
#include "./boundary/boundary_rim_is_flap.hpp"
#include "./boundary_rims.hpp"
#include "./face_membership_like.hpp"
#include "./make_face_membership.hpp"
#include "./policy/face_membership.hpp"
#include <cstddef>

namespace tf {

/// @ingroup topology_analysis
/// @brief Check if a mesh has a boundary rim that is a hole.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @return `true` if a boundary rim is a hole.
template <typename Policy, typename Policy1>
auto has_holes(const tf::faces<Policy> &faces,
               const tf::face_membership_like<Policy1> &fm) -> bool {
  auto rims = tf::make_boundary_rims(faces, fm);
  return tf::parallel_contains(
      tf::make_sequence_range(rims.size()),
      [&faces, &fm, &rims](std::size_t rim) {
        return !tf::topology::boundary_rim_is_flap(faces, fm, rims, rim);
      },
      tf::checked);
}

/// @ingroup topology_analysis
/// @brief Check if a mesh has a boundary rim that is a hole.
///
/// Convenience overload that builds face membership internally if not
/// provided via policy.
///
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @return `true` if a boundary rim is a hole.
template <typename Policy>
auto has_holes(const tf::polygons<Policy> &polygons) -> bool {
  if constexpr (tf::has_face_membership_policy<Policy>) {
    return tf::has_holes(polygons.faces(), polygons.face_membership());
  } else {
    auto fm = tf::make_face_membership(polygons);
    return has_holes(polygons | tf::tag(fm));
  }
}

} // namespace tf
