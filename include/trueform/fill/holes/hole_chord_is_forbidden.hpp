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
#include "../../core/faces.hpp"
#include "../../core/polygons.hpp"
#include "../../topology/face_membership_like.hpp"
#include "./hole_face_carries_edge.hpp"
#include "./hole_rim.hpp"
#include <cstddef>

namespace tf::fill {

/// Whether a non-rim chord between two points is forbidden.
///
/// Two facts forbid one: the mesh already winds that edge, so a patch using
/// it would make it three-incident; or the two ends share a carrying
/// triangle, so the canonical subdivision of that triangle could later join
/// them itself. The second is a discrete ticket join — original corner
/// membership for a vertex, the inherited carrier for a split — and it is
/// deliberately stronger than testing the edges the plan actually creates.
///
/// A rim subedge is the intentional exception and is never asked about: it
/// is the stitch, one host incidence against one patch incidence.
///
/// Each end is named the same way everywhere: its flat identity, and the
/// carrying triangle it inherits — `-1` for an original vertex and for a
/// minted point the host never carried. This is the one producer of the
/// verdict; the rim, the planar product and a refined patch all ask it.
template <typename FacesPolicy, typename MembershipPolicy, typename Index>
auto hole_chord_is_forbidden(
    const tf::faces<FacesPolicy> &faces,
    const tf::face_membership_like<MembershipPolicy> &fm, Index n_points,
    Index flat_a, Index carrier_a, Index flat_b, Index carrier_b) -> bool {
  const bool a_minted = flat_a >= n_points;
  const bool b_minted = flat_b >= n_points;

  if (a_minted && b_minted)
    return carrier_a != Index(-1) && carrier_a == carrier_b;
  if (a_minted || b_minted) {
    const Index carrier = a_minted ? carrier_a : carrier_b;
    const Index vertex = a_minted ? flat_b : flat_a;
    return carrier != Index(-1) &&
           tf::fill::hole_face_has_corner(faces, carrier, vertex);
  }

  for (auto face : fm[std::size_t(flat_a)])
    if (tf::fill::hole_face_carries_edge(faces, face, flat_a, flat_b))
      return true;
  return false;
}

/// The verdict for two prepared positions of a rim, which name their own
/// flat identities and carriers.
template <typename Policy, typename MembershipPolicy, typename Index,
          typename Int, typename RealT>
auto hole_chord_is_forbidden(
    const tf::polygons<Policy> &polygons,
    const tf::face_membership_like<MembershipPolicy> &fm,
    const tf::fill::hole_rim<Index, Int, RealT> &rim, Index a, Index b)
    -> bool {
  const auto carrier_of = [&rim](Index position) {
    return tf::fill::hole_position_is_split(rim, position)
               ? rim.edges[std::size_t(position)].face
               : Index(-1);
  };
  return tf::fill::hole_chord_is_forbidden(
      polygons.faces(), fm, Index(polygons.points().size()),
      rim.corners[std::size_t(a)], carrier_of(a), rim.corners[std::size_t(b)],
      carrier_of(b));
}

} // namespace tf::fill
