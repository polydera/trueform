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
#include "../../core/buffer.hpp"
#include "../../topology/cdt/dt3/find_tet_site_within.hpp"
#include "../../topology/cdt/dt3/recovery/open_tet_stamps.hpp"
#include "../../topology/cdt/dt3/recovery/tet_stamps.hpp"
#include "../../topology/cdt/dt3/tet_position.hpp"
#include "../../topology/cdt/dt3/tet_site_position.hpp"
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// What a created point elects: the site it names, `k_none` where it names
/// none, and whether the site is an earlier mint it is fused into.
template <typename Index> struct tet_created_seat {
  Index site;
  bool fused = false;
};

/// THE ELECTION of a drain's created point at `p`, on the frame: the site
/// standing at `p`, a corner of `cell`, the cell holding it. Where none
/// does, a site standing a created identity the drain's authority holds for
/// the same pair of input faces (@ref tet_identity_authority::minted_for)
/// in `p`'s cell of the grid of side `reach`, the input quantum on the
/// frame, is one the input cannot tell from it (@ref
/// tf::topology::cdt::dt3::tet_shares_reach_cell), and the point is that site:
/// of every such site the smallest slot survives, the point fused into it at
/// its position
/// (@ref tf::topology::cdt::dt3::find_tet_site_within). `stamps` and `flood`
/// are the caller's scratch. O(the cells meeting the box of `reach`).
template <typename DT, typename Authority>
auto elect_tet_created_site(
    const DT &dt,
    const tf::topology::cdt::dt3::tet_position<typename DT::coord_type> &p,
    typename DT::index_type cell, const Authority &authority,
    const std::array<typename DT::index_type, 2> &pair,
    typename DT::coord_type reach, tf::topology::cdt::dt3::tet_stamps &stamps,
    tf::buffer<typename DT::index_type> &flood)
    -> tet_created_seat<typename DT::index_type> {
  namespace dt3 = tf::topology::cdt::dt3;
  tet_created_seat<typename DT::index_type> seat{DT::k_none};
  if (cell == DT::k_none)
    return seat;
  const auto lanes = dt3::tet_site_lanes_of(dt.owner());
  for (auto site : dt.tets()[std::size_t(cell)])
    if (lanes.position(site) == p)
      seat.site = site;
  if (seat.site != DT::k_none || pair[0] < 0)
    return seat;
  dt3::open_tet_stamps(stamps, dt.tets().size());
  seat.site = dt3::find_tet_site_within(
      dt, p, cell, reach,
      [&authority, &pair](typename DT::index_type site) {
        return authority.minted_for(site, pair);
      },
      [&stamps](std::size_t c) { return dt3::take_tet_stamp(stamps, c); },
      flood);
  seat.fused = seat.site != DT::k_none;
  return seat;
}

} // namespace tf::tetmesh
