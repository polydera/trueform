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
#include "../../../core/buffer.hpp"
#include "./tet_position.hpp"
#include "./tet_site_position.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Visit every site in `p`'s cell of the reach grid (@ref
/// tet_shares_reach_cell), as `visit(site)`, once per cell holding it. The
/// sites read are the corners of the finite cells, flooded from `cell`, the
/// cell holding `p`, whose corners' floors, each raised by one, span a box
/// meeting the one of `reach` around `p`'s unit cell, which holds every cell
/// meeting it; a corner's floor reads first, its position only where the
/// floor's unit cell meets that box. `take(c)` is true the first time cell `c`
/// is asked of it, and `flood` is the caller's scratch. O(the cells meeting
/// that box).
template <typename DT, typename Visit, typename Take>
auto for_each_tet_site_within(const DT &dt,
                              const tet_position<typename DT::coord_type> &p,
                              typename DT::index_type cell,
                              typename DT::coord_type reach, const Visit &visit,
                              const Take &take,
                              tf::buffer<typename DT::index_type> &flood)
    -> void {
  using Index = typename DT::index_type;
  using T1 = typename tet_reach_int<typename DT::coord_type>::type;
  if (cell == DT::k_none)
    return;
  const auto lanes = tet_site_lanes_of(dt.owner());
  const auto cells = dt.tets();
  const auto neighbors = dt.neighbors();
  const T1 r = T1(reach);
  const auto offset = [&lanes, &p](Index site, std::size_t k) {
    return T1(lanes.floors[std::size_t(site)].pt[k]) - T1(p.floor[k]);
  };
  const auto within_floor = [&offset, &p, r](Index site) {
    for (std::size_t k = 0; k < 3; ++k) {
      const T1 lo = offset(site, k);
      if (lo > T1(p.residual[k] != 0) + r || lo + T1(1) < -r)
        return false;
    }
    return true;
  };
  const auto meets = [&cells, &offset, &p, r](Index c) {
    const auto &corners = cells[std::size_t(c)];
    for (std::size_t k = 0; k < 3; ++k) {
      T1 lo = offset(corners[0], k), hi = lo;
      for (std::size_t i = 1; i < 4; ++i) {
        const T1 x = offset(corners[i], k);
        lo = x < lo ? x : lo;
        hi = x > hi ? x : hi;
      }
      if (lo > T1(p.residual[k] != 0) + r || hi + T1(1) < -r)
        return false;
    }
    return true;
  };
  flood.clear();
  take(std::size_t(cell));
  flood.push_back(cell);
  for (std::size_t head = 0; head < flood.size(); ++head) {
    const Index c = flood[head];
    for (auto site : cells[std::size_t(c)])
      if (within_floor(site) &&
          tet_shares_reach_cell(lanes.position(site), p, reach))
        visit(site);
    for (std::size_t j = 0; j < 4; ++j) {
      const Index next = neighbors[std::size_t(c)][j];
      if (next != DT::k_none && take(std::size_t(next)) && meets(next))
        flood.push_back(next);
    }
  }
}

} // namespace tf::topology::cdt::dt3
