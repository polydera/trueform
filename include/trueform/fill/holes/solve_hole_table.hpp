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
#include "../../core/polygons.hpp"
#include "../../topology/face_membership_like.hpp"
#include "../hole_fill_status.hpp"
#include "./compact_hole_table_facets.hpp"
#include "./gather_hole_facets.hpp"
#include "./hole_rim.hpp"
#include "./hole_table.hpp"
#include "./order_hole_table_states.hpp"
#include "./prepare_hole_table_candidates.hpp"
#include "./state_hole_table_areas.hpp"
#include "./state_hole_table_bottlenecks.hpp"
#include "./state_hole_table_facets.hpp"
#include <array>
#include <cstddef>

namespace tf::fill {

/// Solve one protected rim over the complex it was protected against.
///
/// The state is `(i, j, a)`: the chord `i j` read against the outer triangle
/// already fixed on its far side, which the root reads against the wrap
/// edge's own carrying triangle. Choosing the apex `k` inside charges the
/// angle across the access chord, and the host angle of a boundary child
/// only the parent can see; the two children `(i, k, j)` and `(k, j, i)`
/// carry the rest. The right child's outer apex is `i`.
///
/// Stage one minimizes the largest charged angle; stage two solves every
/// state AFRESH under the root's own threshold and minimizes area beneath
/// it. A candidate is admitted or refused on its own — an inadmissible one
/// never poisons its siblings — and a state with no admissible candidate is
/// invalid, an invalid root being the rim's refusal.
template <typename Policy, typename MembershipPolicy, typename Index,
          typename Int, typename RealT>
auto solve_hole_table(
    const tf::polygons<Policy> &polygons,
    const tf::face_membership_like<MembershipPolicy> &fm,
    const tf::fill::hole_rim<Index, Int, RealT> &rim,
    const tf::fill::hole_table_scratch<Index, Int, RealT> &complex,
    tf::fill::hole_table<Index> &table,
    tf::buffer<std::array<Index, 3>> &triangles) -> tf::hole_fill_status {
  const Index n = Index(rim.size());

  tf::fill::gather_hole_facets(complex.dt, complex.position_of_site,
                               table.owned, table.facets);
  tf::fill::state_hole_table_facets(polygons, fm, rim, n, table);
  tf::fill::compact_hole_table_facets(table);

  const std::size_t kept = table.facets.size();
  const std::size_t states = 2 * kept + 1;

  tf::fill::prepare_hole_table_candidates(table);
  tf::fill::order_hole_table_states(n, table);

  table.bottleneck.allocate(states);
  table.total.allocate(states);
  table.choice.allocate(states);
  table.valid.allocate(states);
  tf::fill::state_hole_table_bottlenecks(rim, n, table);

  const auto root = Index(2 * kept);
  if (!table.valid[std::size_t(root)])
    return tf::hole_fill_status::refused_table;
  tf::fill::state_hole_table_areas(rim, n, table.bottleneck[std::size_t(root)],
                                   table);

  if (table.choice[std::size_t(root)] < 0)
    return tf::hole_fill_status::refused_table;

  table.stack.clear();
  table.stack.push_back(root);
  while (table.stack.size()) {
    const Index state = table.stack.back();
    table.stack.erase_till_end(table.stack.end() - 1);
    const auto facet = std::size_t(table.choice[std::size_t(state)]);
    const auto corners = table.facets[facet];
    triangles.push_back({rim.corners[std::size_t(corners[2])],
                         rim.corners[std::size_t(corners[1])],
                         rim.corners[std::size_t(corners[0])]});
    for (std::size_t side = 0; side < 2; ++side)
      if (corners[side + 1] != corners[side] + 1)
        table.stack.push_back(Index(2 * facet + side));
  }
  return tf::hole_fill_status::filled;
}

} // namespace tf::fill
