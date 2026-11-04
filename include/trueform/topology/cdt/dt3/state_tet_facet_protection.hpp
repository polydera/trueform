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
#include "./reserve_tet_protection.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// State whether the facet of `cell` opposite `slot` is protected, and the
/// face owning it, `none` for no face, on both cells that hold it, growing
/// the lanes with unprotected rows as it reaches; an unprotected facet owns
/// nothing.
template <typename Owner>
auto state_tet_facet_protection(Owner &owner, typename Owner::index_type cell,
                                std::size_t slot, bool on,
                                typename Owner::index_type face) -> void {
  reserve_tet_protection(owner);
  face = on ? face : Owner::none;
  auto &slots = owner._protected_slots;
  auto &owners = owner._owners;
  const auto mate = owner._neighbors[std::size_t(cell)][slot];
  std::size_t back = 0;
  while (owner._neighbors[std::size_t(mate)][back] != cell)
    ++back;
  const auto bit = [on](std::uint8_t &byte, std::size_t at) {
    if (on)
      byte = std::uint8_t(byte | (1u << at));
    else
      byte = std::uint8_t(byte & ~(1u << at));
  };
  bit(slots[std::size_t(cell)], slot);
  bit(slots[std::size_t(mate)], back);
  owners[4 * std::size_t(cell) + slot] = face;
  owners[4 * std::size_t(mate) + back] = face;
}

} // namespace tf::topology::cdt::dt3
