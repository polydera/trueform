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
#include "../tet_facet_is_protected.hpp"
#include "../tet_facet_key.hpp"
#include "../tet_shared_owners.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Whether face `face` holds the facet of ticket `4 cell + slot`: the facet
/// is protected and the face owns it, alone or in its stack.
template <typename Owner>
auto tet_face_holds_facet(const Owner &owner, typename Owner::index_type ticket,
                          typename Owner::index_type face) -> bool {
  using Index = typename Owner::index_type;
  const Index cell = Index(std::size_t(ticket) / 4);
  const std::size_t slot = std::size_t(ticket) % 4;
  if (!tet_facet_is_protected(owner, cell, slot))
    return false;
  const Index who = owner._owners[std::size_t(ticket)];
  if (who == face)
    return true;
  if (who != Owner::shared_owner)
    return false;
  const auto stack = tet_shared_owners(
      owner, tet_facet_key(owner._corners[std::size_t(cell)], slot));
  for (auto at = stack.first; at != stack.second; ++at)
    if (at->face == face)
      return true;
  return false;
}

} // namespace tf::topology::cdt::dt3
