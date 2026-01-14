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
#include "../core/static_size.hpp"
#include "./face_membership.hpp"
#include "./manifold_edge_link.hpp"
#include "./policy/face_membership.hpp"

namespace tf {

/// @ingroup topology_connectivity
/// @brief Create manifold edge link from polygons.
///
/// Convenience function that builds and returns a manifold_edge_link structure.
/// Automatically builds face_membership if not already tagged on the polygons.
/// The index type and face size are automatically deduced from the polygons.
///
/// @tparam Index The index type (auto-deduced if not specified).
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @return A manifold_edge_link structure containing edge connectivity.
template <typename Index = tf::none_t, typename Policy>
auto make_manifold_edge_link(const tf::polygons<Policy> &polygons) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    return make_manifold_edge_link<
        std::decay_t<decltype(polygons.faces()[0][0])>>(polygons);
  } else {
    constexpr auto NGon = tf::static_size_v<decltype(polygons.faces()[0])>;
    if constexpr (tf::has_face_membership_policy<Policy>) {
      tf::manifold_edge_link<Index, NGon> mel;
      mel.build(polygons.faces(), polygons.face_membership());
      return mel;
    } else {
      tf::face_membership<Index> fm;
      fm.build(polygons);
      tf::manifold_edge_link<Index, NGon> mel;
      mel.build(polygons.faces(), fm);
      return mel;
    }
  }
}

} // namespace tf
