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
#include "../core/area.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/faces.hpp"
#include "../core/polygons.hpp"
#include "../core/static_size.hpp"
#include "../core/views/constant.hpp"
#include "../core/views/mapped_range.hpp"
#include "./face_membership.hpp"
#include "./manifold_edge_link.hpp"
#include "./manifold_edge_link_like.hpp"
#include "./orientation/orient_faces_consistently.hpp"
#include "./policy/manifold_edge_link.hpp"
#include <type_traits>

namespace tf {

/// @ingroup topology_analysis
/// @brief Orient faces consistently with uniform weights.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The manifold edge link policy type.
/// @param faces The faces range (modified in place).
/// @param link The manifold edge link built from exactly those face slots.
/// @return `true` when every component is orientable and now consistent.
template <typename Policy, typename Policy1>
auto orient_faces_consistently(tf::faces<Policy> &faces,
                               const tf::manifold_edge_link_like<Policy1> &link)
    -> bool {
  return topology::orient_faces_consistently(
      faces, link, tf::make_constant_range(1, faces.size()));
}

/// @ingroup topology_analysis
/// @overload
template <typename Policy, typename Policy1>
auto orient_faces_consistently(tf::faces<Policy> &&faces,
                               const tf::manifold_edge_link_like<Policy1> &link)
    -> bool {
  return tf::orient_faces_consistently(faces, link);
}

/// @ingroup topology_analysis
/// @brief Orient faces consistently in a polygons range.
///
/// Uses face area as weights for voting; an integral coordinate type votes by
/// face count instead, its squared area not fitting the lattice it is measured
/// on. Builds manifold edge link internally if not provided via policy. Every
/// face's vertex set, arity and position are preserved; only cyclic direction
/// may change. The manifold-edge component is the carrier: every orientable
/// component is consistent after this one call, and a component whose parity
/// contradicts is left exactly as it was.
///
/// A reversal permutes a face's edge slots, so a manifold edge link supplied
/// through the policy describes the winding it was built from and not the one
/// this call leaves behind.
///
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range (modified in place).
/// @return `true` when every component is orientable and now consistent.
template <typename Policy>
auto orient_faces_consistently(tf::polygons<Policy> &polygons) -> bool {
  auto orient = [&](const auto &link) {
    if constexpr (std::is_integral_v<tf::coordinate_type<Policy>>)
      return topology::orient_faces_consistently(
          polygons.faces(), link,
          tf::make_constant_range(1, polygons.faces().size()));
    else
      return topology::orient_faces_consistently(
          polygons.faces(), link,
          tf::make_mapped_range(
              polygons, [](const auto &poly) { return tf::area2(poly); }));
  };

  if constexpr (tf::has_manifold_edge_link_policy<Policy>)
    return orient(polygons.manifold_edge_link());
  else {
    using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
    tf::face_membership<Index> fm;
    fm.build(polygons);
    tf::manifold_edge_link<Index,
                           tf::static_size_v<decltype(polygons.faces()[0])>>
        mel;
    mel.build(polygons.faces(), fm);
    return orient(mel);
  }
}

/// @ingroup topology_analysis
/// @overload
template <typename Policy>
auto orient_faces_consistently(tf::polygons<Policy> &&polygons) -> bool {
  return orient_faces_consistently(polygons);
}
} // namespace tf
