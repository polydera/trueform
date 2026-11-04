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
#include "../find_tet_edge.hpp"
#include "../tet_edge_ref.hpp"
#include "../tet_orient3d_sites.hpp"
#include "../tet_position.hpp"
#include "../tetrahedralization_owner.hpp"

namespace tf::topology::cdt::dt3 {

/// Whether the position `p` lies strictly inside the kernel of the ring of
/// the edge d e: on each ring cell's own side of both its facets opposite d
/// and opposite e, so a split at `p` leaves the edge to the flips its star
/// makes. A ring holding a hull cell holds no kernel.
template <typename Index, typename Coord, typename Int>
auto tet_point_in_edge_kernel(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type d,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type e,
    const tet_position<Coord> &p) -> bool {
  using owner_t = tetrahedralization_owner<Index, Coord, Int>;
  tet_edge_ref<Index> handle{};
  if (!find_tet_edge(owner, d, e, handle))
    return false;
  auto spin = handle;
  do {
    const auto c = tet_edge_corners(owner, spin);
    if (c[2] == owner_t::infinite || c[3] == owner_t::infinite)
      return false;
    for (int side = 0; side < 2; ++side) {
      const Index keep = side ? c[1] : c[0], other = side ? c[0] : c[1];
      const int want = tet_orient3d_sites(owner, keep, c[2], c[3], other);
      if (!want || tet_orient3d_toward(owner, keep, c[2], c[3], p) != want)
        return false;
    }
    spin = tet_edge_next(owner, spin);
  } while (spin.cell != handle.cell);
  return true;
}

} // namespace tf::topology::cdt::dt3
