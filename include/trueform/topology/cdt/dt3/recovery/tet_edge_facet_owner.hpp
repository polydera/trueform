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
#include "../tet_facet_is_protected.hpp"
#include "../tet_facet_key.hpp"
#include "../tet_shared_owners.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The face owning a protected facet about the edge from `d` to `e`, read
/// round the edge's ring from the handle @ref find_tet_edge gives, `star`
/// its walk's list; a stack answers its first owner. `-1` where the edge
/// stands in no protected facet or the complex holds no such edge. Every
/// cell is admitted before any lane of it is read, so a ring reaching
/// another region stops the task and answers none (@ref admit_tet_row).
template <typename Owner, typename Index>
auto tet_edge_facet_owner(const Owner &owner, typename Owner::index_type d,
                          typename Owner::index_type e,
                          tet_edge_star<typename Owner::index_type> &star,
                          tet_round_state<Index> &round) -> Index {
  tet_edge_ref<Index> handle{};
  if (!find_tet_edge(owner, d, e, handle, star, round))
    return Index(-1);
  auto spin = handle;
  do {
    const std::size_t slot = tet_edge_oppo(spin);
    if (tet_facet_is_protected(owner, spin.cell, slot)) {
      const Index who = owner._owners[4 * std::size_t(spin.cell) + slot];
      if (who == Owner::shared_owner) {
        const auto stack = tet_shared_owners(
            owner, tet_facet_key(owner._corners[std::size_t(spin.cell)], slot));
        if (stack.first != stack.second)
          return stack.first->face;
      } else if (who != Owner::none)
        return who;
    }
    spin = admitted_tet_edge_next(owner, round, spin);
    if (round.stopped)
      return Index(-1);
  } while (spin.cell != handle.cell);
  return Index(-1);
}

/// The same outside a round, where every cell is the one task's.
template <typename Owner>
auto tet_edge_facet_owner(const Owner &owner, typename Owner::index_type d,
                          typename Owner::index_type e,
                          tet_edge_star<typename Owner::index_type> &star) ->
    typename Owner::index_type {
  tet_round_state<typename Owner::index_type> closed;
  return tet_edge_facet_owner(owner, d, e, star, closed);
}

} // namespace tf::topology::cdt::dt3
