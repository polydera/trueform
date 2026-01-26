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
#include "./face_membership.hpp"
#include "./policy/face_membership.hpp"
#include "./vertex_link.hpp"

namespace tf {

/// @ingroup topology_connectivity
/// @brief Create vertex link from polygons.
///
/// Convenience function that builds and returns a vertex_link structure.
/// Automatically builds face_membership if not already tagged on the polygons.
/// The index type is automatically deduced from the polygons unless specified.
///
/// @tparam Index The index type (auto-deduced if not specified).
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @return A vertex_link structure containing 1-ring neighborhoods.
template <typename Index = tf::none_t, typename Policy>
auto make_vertex_link(const tf::polygons<Policy> &polygons) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    return make_vertex_link<std::decay_t<decltype(polygons.faces()[0][0])>>(
        polygons);
  } else {
    if constexpr (tf::has_face_membership_policy<Policy>) {
      tf::vertex_link<Index> vl;
      vl.build(polygons.faces(), polygons.face_membership());
      return vl;
    } else {
      tf::face_membership<Index> fm;
      fm.build(polygons);
      tf::vertex_link<Index> vl;
      vl.build(polygons.faces(), fm);
      return vl;
    }
  }
}

/// @ingroup topology_connectivity
/// @brief Create vertex link from segments.
///
/// Convenience function that builds and returns a vertex_link structure
/// from segment connectivity. The index type is automatically deduced
/// from the segments unless specified.
///
/// @tparam Index The index type (auto-deduced if not specified).
/// @tparam Policy The segments policy type.
/// @param segments The segments range.
/// @param eo The edge orientation mode (default: bidirectional).
/// @return A vertex_link structure containing vertex neighborhoods.
template <typename Index = tf::none_t, typename Policy>
auto make_vertex_link(
    const tf::segments<Policy> &segments,
    tf::edge_orientation eo = tf::edge_orientation::bidirectional) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    return make_vertex_link<std::decay_t<decltype(segments.edges()[0][0])>>(
        segments, eo);
  } else {
    tf::vertex_link<Index> vl;
    vl.build(segments, eo);
    return vl;
  }
}

} // namespace tf
