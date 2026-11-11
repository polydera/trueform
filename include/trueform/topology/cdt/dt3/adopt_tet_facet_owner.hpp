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
#include "./share_tet_facet.hpp"
#include "./state_tet_facet_protection.hpp"
#include "./tet_round_state.hpp"
#include "./tet_facet_key.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Give the protected facet of `cell` opposite `slot` the owner `face`,
/// covering `layers` of it: an unowned facet takes it, one another face owns
/// becomes a stack of both, a stack takes it as one more row; a face owning
/// it with more than one layer holds it as a stack's row, alone or not, so
/// the row can say how many. A `face` of none changes nothing.
template <typename Owner, typename Index>
auto adopt_tet_facet_owner(Owner &owner, typename Owner::index_type cell,
                           std::size_t slot, typename Owner::index_type face,
                           typename Owner::index_type layers,
                           tet_round_state<Index> &round) -> void {
  if (face == Owner::none)
    return;
  const Index was = owner._owners[4 * std::size_t(cell) + slot];
  if (was == face && layers == 1)
    return;
  if (was == Owner::none && layers == 1) {
    state_tet_facet_protection(owner, cell, slot, true, face, round);
    return;
  }
  const auto key = tet_facet_key(owner._corners[std::size_t(cell)], slot);
  if (was != Owner::shared_owner) {
    if (was != Owner::none && was != face)
      share_tet_facet(owner, key, was, Index(1), round);
    state_tet_facet_protection(owner, cell, slot, true, Owner::shared_owner,
                               round);
  }
  share_tet_facet(owner, key, face, layers, round);
}

/// The same outside a round, where every row is the one task's.
template <typename Owner>
auto adopt_tet_facet_owner(Owner &owner, typename Owner::index_type cell,
                           std::size_t slot, typename Owner::index_type face,
                           typename Owner::index_type layers = 1) -> void {
  tet_round_state<typename Owner::index_type> closed;
  adopt_tet_facet_owner(owner, cell, slot, face, layers, closed);
}

} // namespace tf::topology::cdt::dt3
