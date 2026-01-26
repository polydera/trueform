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
#include "./policy/manifold_edge_link.hpp"

namespace tf {

/// @ingroup topology_analysis
/// @brief Check if a mesh has no boundary edges.
///
/// Returns `true` if every edge in the mesh is shared by exactly two faces,
/// meaning the mesh has no holes or open boundaries. A closed mesh is
/// watertight and encloses a volume.
///
/// Builds manifold edge link internally if not provided via policy.
///
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @return `true` if the mesh is closed (no boundary edges).
template <typename Policy>
auto is_closed(const tf::polygons<Policy> &polygons) {
  if constexpr (tf::has_manifold_edge_link_policy<Policy>) {
    for (const auto &hs : polygons.manifold_edge_link())
      for (const auto &e : hs)
        if (e.is_boundary())
          return false;
    return true;
  } else {
    using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
    tf::face_membership<Index> fm;
    fm.build(polygons);
    tf::manifold_edge_link<Index,
                           tf::static_size_v<decltype(polygons.faces()[0])>>
        mel;
    mel.build(polygons.faces(), fm);
    return is_closed(polygons | tf::tag(mel));
  }
}
} // namespace tf
