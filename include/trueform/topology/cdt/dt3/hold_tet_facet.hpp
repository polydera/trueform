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
#include "./adopt_tet_facet_owner.hpp"
#include "./state_tet_facet_protection.hpp"
#include "./tet_facet_is_protected.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Hold the facet of ticket `4 cell + slot` as face `face`'s child: an
/// unprotected facet is protected with it as owner, a protected one adopts
/// it, into its stack when other faces hold it.
template <typename Owner>
auto hold_tet_facet(Owner &owner, typename Owner::index_type ticket,
                    typename Owner::index_type face) -> void {
  using Index = typename Owner::index_type;
  const Index cell = Index(std::size_t(ticket) / 4);
  const std::size_t slot = std::size_t(ticket) % 4;
  if (tet_facet_is_protected(owner, cell, slot)) {
    adopt_tet_facet_owner(owner, cell, slot, face);
    return;
  }
  state_tet_facet_protection(owner, cell, slot, true, face);
  ++owner._n_protected;
}

} // namespace tf::topology::cdt::dt3
