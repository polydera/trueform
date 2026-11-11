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
#include "../../../../../core/buffer.hpp"
#include "../../record_tet_written_row.hpp"
#include "../../share_tet_facet.hpp"
#include "../../tet_round_state.hpp"
#include "../../tet_row_pool.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// What the regions wrote of the complex's own tables, merged IN REGION
/// ORDER after the round's barrier, which makes the merge the one producer
/// of each of them: the written record (whose dedupe is its own), the killed
/// edges, the facets the holds shared — inserted into the sorted table in
/// region order, which is the one place it is moved — the held count, the
/// flip counts, and the dead rows — the pool rows a region took
/// left the dead behind them, so each region's take is subtracted and what
/// it did not take stays dead for the reclaim. O(written + killed +
/// regions).
template <typename Owner, typename States>
auto merge_tet_round_states(Owner &owner, const tf::buffer<tet_row_pool> &pools,
                            States &states) -> void {
  for (std::size_t p = 0; p < pools.size(); ++p) {
    auto &state = states[p];
    if (owner._record_written)
      for (auto row : state.written)
        record_tet_written_row(owner, row);
    if (owner._record_killed)
      for (const auto &edge : state.killed)
        owner._killed.push_back(edge);
    for (const auto &row : state.shared)
      share_tet_facet(owner, row.key, row.face, row.layers);
    owner._n_held += state.held;
    owner._stats.flips += state.flips;
    owner._stats.edge_flips += state.edge_flips;
    owner._stats.quad_flips += state.quad_flips;
    owner._n_dead += state.dead;
    owner._n_dead -= state.rows.next - pools[p].next;
  }
}

} // namespace tf::topology::cdt::dt3
