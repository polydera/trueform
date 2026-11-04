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
#include "./tet_facet_key.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Give the protected facet of `cell` opposite `slot` the owner `face`: an
/// unowned facet takes it, one another face owns becomes a stack of both,
/// a stack takes it as one more row; a facet the face owns already or a
/// `face` of none changes nothing.
template <typename Owner>
auto adopt_tet_facet_owner(Owner &owner, typename Owner::index_type cell,
                           std::size_t slot, typename Owner::index_type face)
    -> void {
  using Index = typename Owner::index_type;
  if (face == Owner::none)
    return;
  const Index was = owner._owners[4 * std::size_t(cell) + slot];
  if (was == face)
    return;
  if (was == Owner::none) {
    state_tet_facet_protection(owner, cell, slot, true, face);
    return;
  }
  const auto key = tet_facet_key(owner._corners[std::size_t(cell)], slot);
  if (was != Owner::shared_owner) {
    share_tet_facet(owner, key, was);
    state_tet_facet_protection(owner, cell, slot, true, Owner::shared_owner);
  }
  share_tet_facet(owner, key, face);
}

} // namespace tf::topology::cdt::dt3
