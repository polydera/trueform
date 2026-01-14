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
#include "../core/none.hpp"
#include "./face_link.hpp"
#include "./face_membership.hpp"
#include "./policy/face_membership.hpp"

namespace tf {

/// @ingroup topology_connectivity
/// @brief Create face link from polygons.
///
/// Convenience function that builds and returns a face_link structure.
/// Automatically builds face_membership if not already tagged on the polygons.
/// The index type is automatically deduced from the polygons unless specified.
///
/// @tparam Index The index type (auto-deduced if not specified).
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @return A face_link structure containing face adjacency through shared edges.
template <typename Index = tf::none_t, typename Policy>
auto make_face_link(const tf::polygons<Policy> &polygons) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    return make_face_link<std::decay_t<decltype(polygons.faces()[0][0])>>(
        polygons);
  } else {
    if constexpr (tf::has_face_membership_policy<Policy>) {
      tf::face_link<Index> fl;
      fl.build(polygons.faces(), polygons.face_membership());
      return fl;
    } else {
      tf::face_membership<Index> fm;
      fm.build(polygons);
      tf::face_link<Index> fl;
      fl.build(polygons.faces(), fm);
      return fl;
    }
  }
}

} // namespace tf
