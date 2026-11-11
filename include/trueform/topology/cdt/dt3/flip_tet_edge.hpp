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
#include "./tet_orient3d_sites.hpp"
#include "./admitted_tet_edge.hpp"
#include "./tet_round_state.hpp"
#include "./tet_state.hpp"
#include "./write_tet.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Flip the edge `p q` 3-2, `p q` the facet edge `edge` of the facet of
/// `cell` opposite `slot`: `cell`, with apex `a` at `slot` over the facet
/// `p q r`, the finite cell across it, with apex `w`, and the finite cell
/// `a w p q` beyond both become the two cells `a w r p` and `a w r q`,
/// written into the first two rows, the third killed, linked, the six outer
/// facets' protection carried. The caller has read the ring about `p q` as
/// exactly these three cells, and `p` and `q` strictly on opposite sides of
/// the plane `a w r`, which with the ring is the two cells' positive
/// orientation, and none of the three facets the ring shares is protected.
/// The edge `p q` dies and no other; no facet of any other cell changes.
/// Returns the two rows.
template <typename Owner>
auto flip_tet_edge(Owner &owner, typename Owner::index_type cell,
                   std::size_t slot, std::size_t edge,
                   tet_round_state<typename Owner::index_type> &round)
    -> std::array<typename Owner::index_type, 2> {
  using Index = typename Owner::index_type;
  owner._canonical = false;
  const auto read = [&owner](Index c) {
    const auto row = owner._corners[std::size_t(c)];
    return std::array<Index, 4>{row[0], row[1], row[2], row[3]};
  };
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
  const auto has = [](const std::array<Index, 3> &facet, Index corner) {
    return facet[0] == corner || facet[1] == corner || facet[2] == corner;
  };
  const auto c0 = read(cell);
  const Index a = c0[slot];
  const auto facet = tet_facet(c0, slot);
  const Index p = facet[edge], q = facet[(edge + 1) % 3],
              r = facet[(edge + 2) % 3];
  const Index other = owner._neighbors[std::size_t(cell)][slot];
  if (!admit_tet_flip_closure(owner, round, cell) ||
      !admit_tet_flip_closure(owner, round, other))
    return {Owner::none, Owner::none};
  const auto c1 = read(other);
  std::size_t wslot = 0;
  while (wslot < 4 && (c1[wslot] == p || c1[wslot] == q || c1[wslot] == r))
    ++wslot;
  if constexpr (tf::core::verification)
    tf::core::verify(wslot < 4, "the mate holds an apex off the facet");
  const Index w = c1[wslot];
  const Index third = owner._neighbors[std::size_t(cell)][slot_of(c0, r)];
  if constexpr (tf::core::verification)
    tf::core::verify(third ==
                         owner._neighbors[std::size_t(other)][slot_of(c1, r)],
                     "three cells ring the flipped edge");
  const std::array<Index, 3> ring{cell, other, third};
  const std::array<std::array<Index, 4>, 3> corners{c0, c1, read(third)};
  struct side {
    Index cell;
    std::size_t slot;
    bool held;
    Index owner;
  };
  std::array<std::array<side, 2>, 3> outer{};
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t x = 0; x < 2; ++x) {
      const std::size_t s = slot_of(corners[i], x == 0 ? q : p);
      const Index n = owner._neighbors[std::size_t(ring[i])][s];
      outer[i][x] = {n, back_slot(n, ring[i]),
                     tet_facet_is_protected(owner, ring[i], s),
                     tet_facet_is_protected(owner, ring[i], s)
                         ? owner._owners[4 * std::size_t(ring[i]) + s]
                         : Owner::none};
    }
  if constexpr (tf::core::verification)
    tf::core::verify(!tet_facet_is_protected(owner, cell, slot) &&
                         !tet_facet_is_protected(owner, cell, slot_of(c0, r)) &&
                         !tet_facet_is_protected(owner, other, slot_of(c1, r)),
                     "a flip removes no protected facet");
  const int sign = tet_orient3d_sites(owner, a, w, r, p);
  if constexpr (tf::core::verification)
    tf::core::verify(sign != 0, "a flip makes no flat cell");
  const std::array<Index, 2> rows{cell, other};
  if (!admit_tet_site_patch(round, a) || !admit_tet_site_patch(round, w))
    return {Owner::none, Owner::none};
  if (sign > 0) {
    write_tet(owner, rows[0], a, w, r, p, round);
    write_tet(owner, rows[1], w, a, r, q, round);
  } else {
    write_tet(owner, rows[0], w, a, r, p, round);
    write_tet(owner, rows[1], a, w, r, q, round);
  }
  owner._states[std::size_t(third)] = tet_state::dead;
  if (std::size_t(third) < owner._protected_slots.size())
    owner._protected_slots[std::size_t(third)] = 0;
  if (round.open())
    ++round.dead;
  else
    ++owner._n_dead;
  for (std::size_t k = 0; k < 2; ++k) {
    const auto made = read(rows[k]);
    auto link = owner._neighbors[std::size_t(rows[k])];
    for (std::size_t t = 0; t < 4; ++t) {
      const auto f = tet_facet(made, t);
      if (!has(f, p) && !has(f, q)) {
        link[t] = rows[1 - k];
        continue;
      }
      const std::size_t i = !has(f, w) ? 0 : !has(f, a) ? 1 : 2;
      const side &o = outer[i][k];
      link[t] = o.cell;
      owner._neighbors[std::size_t(o.cell)][o.slot] = rows[k];
    }
  }
  for (std::size_t k = 0; k < 2; ++k) {
    const auto made = read(rows[k]);
    for (std::size_t t = 0; t < 4; ++t) {
      const auto f = tet_facet(made, t);
      if (!has(f, p) && !has(f, q))
        continue;
      const std::size_t i = !has(f, w) ? 0 : !has(f, a) ? 1 : 2;
      if (outer[i][k].held)
        state_tet_facet_protection(owner, rows[k], t, true, outer[i][k].owner);
    }
  }
  owner._hint = cell;
  owner._delaunay = false;
  if (round.open())
    ++round.edge_flips;
  else
    ++owner._stats.edge_flips;
  return rows;
}

/// The same outside a round.
template <typename Owner>
auto flip_tet_edge(Owner &owner, typename Owner::index_type cell,
                   std::size_t slot, std::size_t edge)
    -> std::array<typename Owner::index_type, 2> {
  tet_round_state<typename Owner::index_type> closed;
  return flip_tet_edge(owner, cell, slot, edge, closed);
}

} // namespace tf::topology::cdt::dt3
