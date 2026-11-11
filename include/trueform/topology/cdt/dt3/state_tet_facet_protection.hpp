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
#include "./reserve_tet_protection.hpp"
#include "./tet_round_state.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// State whether the facet of `cell` opposite `slot` is protected, and the
/// face owning it, `none` for no face, on both cells that hold it, growing
/// the lanes with unprotected rows as it reaches; an unprotected facet owns
/// nothing. The facet was protected when either cell held its bit, so a
/// facet restated on a rewritten row keeps its count through the row
/// across it. THE MATE IS A REACH: the facet's second cell is obtained
/// through the back slot, so a round states nothing where the mate is
/// another region's and the task stops (@ref admit_tet_row); the held count
/// is the region's share while a round runs (@ref tet_round_state).
template <typename Owner, typename Index>
auto state_tet_facet_protection(Owner &owner, typename Owner::index_type cell,
                                std::size_t slot, bool on,
                                typename Owner::index_type face,
                                tet_round_state<Index> &round) -> void {
  if (!admit_tet_row(owner, round,
                     owner._neighbors[std::size_t(cell)][slot]))
    return;
  reserve_tet_protection(owner);
  face = on ? face : Owner::none;
  auto &slots = owner._protected_slots;
  auto &owners = owner._owners;
  const auto mate = owner._neighbors[std::size_t(cell)][slot];
  std::size_t back = 0;
  while (back < 4 && owner._neighbors[std::size_t(mate)][back] != cell)
    ++back;
  if constexpr (tf::core::verification)
    tf::core::verify(back < 4, "the mate names the cell back");
  const auto bit = [on](std::uint8_t &byte, std::size_t at) {
    if (on)
      byte = std::uint8_t(byte | (1u << at));
    else
      byte = std::uint8_t(byte & ~(1u << at));
  };
  const bool was = ((slots[std::size_t(cell)] >> slot) & 1u) ||
                   ((slots[std::size_t(mate)] >> back) & 1u);
  if (was != on) {
    if (round.open())
      round.held += on ? 1 : std::size_t(-1);
    else
      owner._n_held = on ? owner._n_held + 1 : owner._n_held - 1;
  }
  bit(slots[std::size_t(cell)], slot);
  bit(slots[std::size_t(mate)], back);
  owners[4 * std::size_t(cell) + slot] = face;
  owners[4 * std::size_t(mate) + back] = face;
}

/// The same outside a round, where every row is the one task's.
template <typename Owner>
auto state_tet_facet_protection(Owner &owner, typename Owner::index_type cell,
                                std::size_t slot, bool on,
                                typename Owner::index_type face) -> void {
  tet_round_state<typename Owner::index_type> closed;
  state_tet_facet_protection(owner, cell, slot, on, face, closed);
}

} // namespace tf::topology::cdt::dt3
