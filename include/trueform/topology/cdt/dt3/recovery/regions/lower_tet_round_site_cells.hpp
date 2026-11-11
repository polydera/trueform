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
#include "../../tet_round_state.hpp"
#include "../../tet_state.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The site-cell slots the round's regions wrote, lowered to the lowest live
/// cell holding each site: the written rows are a set and a minimum is
/// order-free, so the table the round leaves is the same at any worker
/// count. A site a written row held is held by a live written row, because a
/// flip covers the star it rewrites. O(written).
template <typename Owner, typename States>
auto lower_tet_round_site_cells(Owner &owner, States &states) -> void {
  using Index = typename Owner::index_type;
  if (!owner._record_site_cells)
    return;
  const auto live_written = [&owner, &states](auto &&f) {
    for (std::size_t p = 0; p < states.size(); ++p)
      for (auto row : states[p].written)
        if (owner._states[std::size_t(row)] == tet_state::live)
          for (auto corner : owner._corners[std::size_t(row)])
            f(std::size_t(corner), row);
  };
  live_written([&owner](std::size_t site, Index) {
    const Index at = owner._site_cells.load(site);
    if (at < 0 || owner._states[std::size_t(at)] != tet_state::live)
      owner._site_cells.store(site, Owner::none);
  });
  live_written([&owner](std::size_t site, Index row) {
    owner._site_cells.lower(site, row, Owner::none);
  });
}

} // namespace tf::topology::cdt::dt3
