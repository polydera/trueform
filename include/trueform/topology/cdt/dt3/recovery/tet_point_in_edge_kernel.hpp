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
#include "../admitted_tet_edge.hpp"
#include "../find_tet_edge.hpp"
#include "../tet_edge_ref.hpp"
#include "../tet_edge_star.hpp"
#include "../tet_orient3d_sites.hpp"
#include "../tet_position.hpp"
#include "../tetrahedralization_owner.hpp"

namespace tf::topology::cdt::dt3 {

/// Whether the position `p` lies strictly inside the kernel of the ring of
/// the edge d e: on each ring cell's own side of both its facets opposite d
/// and opposite e, so a split at `p` leaves the edge to the flips its star
/// makes. A ring holding a hull cell holds no kernel. Every cell is admitted
/// before any lane of it is read, so a ring reaching another region stops
/// the task and the answer is no kernel (@ref admit_tet_row).
template <typename Index, typename Coord, typename Int>
auto tet_point_in_edge_kernel(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type d,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type e,
    const tet_position<Coord> &p, tet_edge_star<Index> &star,
    tet_round_state<Index> &round) -> bool {
  using owner_t = tetrahedralization_owner<Index, Coord, Int>;
  tet_edge_ref<Index> handle{};
  if (!find_tet_edge(owner, d, e, handle, star, round))
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
    spin = admitted_tet_edge_next(owner, round, spin);
    if (round.stopped)
      return false;
  } while (spin.cell != handle.cell);
  return true;
}

/// The same outside a round, where every cell is the one task's.
template <typename Index, typename Coord, typename Int>
auto tet_point_in_edge_kernel(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type d,
    typename tetrahedralization_owner<Index, Coord, Int>::index_type e,
    const tet_position<Coord> &p, tet_edge_star<Index> &star) -> bool {
  tet_round_state<Index> closed;
  return tet_point_in_edge_kernel(owner, d, e, p, star, closed);
}

} // namespace tf::topology::cdt::dt3
