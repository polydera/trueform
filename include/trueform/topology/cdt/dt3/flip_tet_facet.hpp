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
#include "../../../core/config/verify.hpp"
#include "./state_tet_facet_protection.hpp"
#include "./tet_facet_is_protected.hpp"
#include "./tet_facets.hpp"
#include "./admitted_tet_edge.hpp"
#include "./tet_round_state.hpp"
#include "./write_tet.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Flip the facet of `cell` opposite `slot` 2-3: `cell`, with apex `a` at
/// `slot`, and the finite cell across the facet, with apex `w`, become the
/// three cells around the edge `a w`, one `(a, p, q, w)` over each facet edge
/// `p q`, written into the two rows and one fresh row, linked, the facets'
/// protection carried. The caller has read `w` strictly inside the facet's
/// cone from `a`, which is the three cells' positive orientation; no edge
/// dies. The complex is no longer Delaunay, and `_delaunay` says so. Returns
/// the three rows in facet-edge order, `a` at slot 0 and `w` at slot 3.
template <typename Owner>
auto flip_tet_facet(Owner &owner, typename Owner::index_type cell,
                    std::size_t slot,
                    tet_round_state<typename Owner::index_type> &round)
    -> std::array<typename Owner::index_type, 3> {
  using Index = typename Owner::index_type;
  owner._canonical = false;
  const auto read = [&owner](Index c) {
    const auto row = owner._corners[std::size_t(c)];
    return std::array<Index, 4>{row[0], row[1], row[2], row[3]};
  };
  const auto c0 = read(cell);
  const Index a = c0[slot];
  const auto facet = tet_facet(c0, slot);
  const Index other = owner._neighbors[std::size_t(cell)][slot];
  if (!admit_tet_flip_closure(owner, round, cell) ||
      !admit_tet_flip_closure(owner, round, other) ||
      !admit_tet_site_patch(round, a))
    return {Owner::none, Owner::none, Owner::none};
  const auto c1 = read(other);
  std::size_t wslot = 0;
  while (wslot < 4 && (c1[wslot] == facet[0] || c1[wslot] == facet[1] ||
                       c1[wslot] == facet[2]))
    ++wslot;
  if constexpr (tf::core::verification)
    tf::core::verify(wslot < 4, "the mate holds an apex off the facet");
  const Index w = c1[wslot];
  const auto slot_of = [](const std::array<Index, 4> &corners, Index corner) {
    std::size_t s = 0;
    while (s < 4 && corners[s] != corner)
      ++s;
    if constexpr (tf::core::verification)
      tf::core::verify(s < 4, "the cell holds the corner");
    return s;
  };
  const auto back_slot = [&owner](Index from, Index to) {
    std::size_t s = 0;
    while (s < 4 && owner._neighbors[std::size_t(from)][s] != to)
      ++s;
    if constexpr (tf::core::verification)
      tf::core::verify(s < 4, "the mate names the cell back");
    return s;
  };
  struct side {
    Index cell;
    std::size_t slot;
    bool held;
    Index owner;
  };
  std::array<side, 3> over{}, under{};
  for (std::size_t i = 0; i < 3; ++i) {
    const Index r = facet[i];
    const std::size_t s0 = slot_of(c0, r), s1 = slot_of(c1, r);
    const Index n0 = owner._neighbors[std::size_t(cell)][s0];
    const Index n1 = owner._neighbors[std::size_t(other)][s1];
    under[i] = {n0, back_slot(n0, cell),
                tet_facet_is_protected(owner, cell, s0),
                tet_facet_is_protected(owner, cell, s0)
                    ? owner._owners[4 * std::size_t(cell) + s0]
                    : Owner::none};
    over[i] = {n1, back_slot(n1, other),
               tet_facet_is_protected(owner, other, s1),
               tet_facet_is_protected(owner, other, s1)
                   ? owner._owners[4 * std::size_t(other) + s1]
                   : Owner::none};
  }
  Index fresh;
  if (round.open()) {
    if constexpr (tf::core::verification)
      tf::core::verify(!round.rows.spent(),
                       "a round reserved a row for every flip of its region");
    fresh = Index(round.rows.next++);
  } else {
    fresh = Index(owner._corners.size());
    owner._corners.reallocate(std::size_t(fresh) + 1);
    owner._neighbors.reallocate(std::size_t(fresh) + 1);
    owner._states.reallocate(std::size_t(fresh) + 1);
  }
  const std::array<Index, 3> rows{cell, other, fresh};
  for (std::size_t i = 0; i < 3; ++i)
    write_tet(owner, rows[i], a, facet[i], facet[(i + 1) % 3], w, round);
  for (std::size_t i = 0; i < 3; ++i) {
    const std::size_t r = (i + 2) % 3;
    auto link = owner._neighbors[std::size_t(rows[i])];
    link[0] = over[r].cell;
    owner._neighbors[std::size_t(over[r].cell)][over[r].slot] = rows[i];
    link[3] = under[r].cell;
    owner._neighbors[std::size_t(under[r].cell)][under[r].slot] = rows[i];
    link[1] = rows[(i + 1) % 3];
    owner._neighbors[std::size_t(rows[(i + 1) % 3])][2] = rows[i];
  }
  for (std::size_t i = 0; i < 3; ++i) {
    const std::size_t r = (i + 2) % 3;
    if (over[r].held)
      state_tet_facet_protection(owner, rows[i], 0, true, over[r].owner);
    if (under[r].held)
      state_tet_facet_protection(owner, rows[i], 3, true, under[r].owner);
  }
  owner._hint = cell;
  owner._delaunay = false;
  if (round.open())
    ++round.flips;
  else
    ++owner._stats.flips;
  return rows;
}

/// The same where no round reserved rows: the flip appends to `owner`.
template <typename Owner>
auto flip_tet_facet(Owner &owner, typename Owner::index_type cell,
                    std::size_t slot)
    -> std::array<typename Owner::index_type, 3> {
  tet_round_state<typename Owner::index_type> closed;
  return flip_tet_facet(owner, cell, slot, closed);
}

} // namespace tf::topology::cdt::dt3
