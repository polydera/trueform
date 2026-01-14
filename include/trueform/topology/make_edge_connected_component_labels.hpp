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
#include "./face_membership.hpp"
#include "./label_connected_components.hpp"
#include "./make_applier.hpp"
#include "./policy/face_link.hpp"

namespace tf {

/// @ingroup topology_components
/// @brief Label edge-connected face components.
///
/// Two faces are in the same component if they share any vertex.
/// This labels each face with its component index.
///
/// Builds face link internally if not provided via policy.
///
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @return A @ref tf::connected_component_labels with per-face labels.
template <typename Policy>
auto make_edge_connected_component_labels(const tf::polygons<Policy> &polygons) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  tf::connected_component_labels<Index> out;
  out.labels.allocate(polygons.size());
  if constexpr (tf::has_face_link_policy<Policy>)
    out.n_components = tf::label_connected_components<Index>(
        out.labels, tf::make_applier(polygons.face_link()));
  else {
    tf::face_membership<Index> fm;
    fm.build(polygons);
    tf::face_link<Index> fl;
    fl.build(polygons.faces(), fm);
    out.n_components = tf::label_connected_components<Index>(
        out.labels, tf::make_applier(fl));
  }
  return out;
}
} // namespace tf
