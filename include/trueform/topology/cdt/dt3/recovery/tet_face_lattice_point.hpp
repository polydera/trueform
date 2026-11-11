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
#include "../../../../exact/meta.hpp"
#include "../../../../exact/orient3d.hpp"
#include "../../../../exact/vertex.hpp"
#include "../tet_deepest_of.hpp"
#include "../tet_position.hpp"
#include "../tet_with_sites.hpp"
#include "../tetrahedralization_owner.hpp"
#include "./tet_plane_lattice_point.hpp"
#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// The point of the face `facet` off its rim near where the edge d e
/// crosses its plane, on the face's lattice `depth` bits below the frame
/// (@ref tet_plane_lattice_point), minted where it stands
/// (@ref tet_minted_position). The decision reads every site at the deepest
/// of `depth`, d and e — the facet's corners are build sites on the frame —
/// and the edge's volumes against the plane with them, so the face's
/// lattice lies as many bits apart there as the decision stands below
/// `depth`.
template <typename Index, typename Coord, typename Int, typename Facet>
auto tet_face_lattice_point(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    const Facet &facet, const std::pair<int, int> &axes,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type d,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type e,
    int depth, tet_position<Coord> &out) -> bool {
  const int decision = std::max(depth, tet_deepest_of(owner, d, e));
  return tet_with_sites(owner, decision, [&](const auto &sites) {
    using sites_t = std::decay_t<decltype(sites)>;
    using lattice = typename sites_t::int_type;
    using T2 = typename tf::exact::meta<lattice>::T2;
    const auto &first = facet[0];
    const auto a = sites[std::size_t(first[0])].pt;
    const auto b = sites[std::size_t(first[1])].pt;
    const auto c = sites[std::size_t(first[2])].pt;
    const T2 od = tf::exact::orient3d_value_scaled<lattice>(
        a, b, c, sites[std::size_t(d)].pt);
    const T2 oe = tf::exact::orient3d_value_scaled<lattice>(
        a, b, c, sites[std::size_t(e)].pt);
    tf::exact::pt3<typename sites_t::coord_type> found;
    if (!tet_plane_lattice_point(sites, facet, axes, d, e, od, oe, found,
                                 decision - depth))
      return false;
    out = tet_minted_position<Coord>(found, 0, decision);
    return true;
  });
}

} // namespace tf::topology::cdt::dt3
