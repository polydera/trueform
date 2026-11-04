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
#include "./tet_orient3d_sites.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Write corners and reset neighbors and protection, preserving orientation
/// while moving the infinity corner to slot three; while site cells are
/// recorded, make a finite cell its corners' site cell, and while rows are
/// recorded, record a finite one. Returns the final slot of c3.
template <typename Owner>
auto write_tet_topology(Owner &owner, typename Owner::index_type tet,
                        typename Owner::index_type c0,
                        typename Owner::index_type c1,
                        typename Owner::index_type c2,
                        typename Owner::index_type c3) -> std::size_t {
  using Index = typename Owner::index_type;
  std::array<Index, 4> corners{c0, c1, c2, c3};
  std::size_t base = 3;
  if (corners[3] != Owner::infinite)
    for (std::size_t i = 0; i < 3; ++i)
      if (corners[i] == Owner::infinite) {
        std::swap(corners[i], corners[3]);
        std::swap(corners[(i + 1) % 3], corners[(i + 2) % 3]);
        base = i;
        break;
      }

  assert(corners[3] == Owner::infinite ||
         tet_orient3d_sites(owner, corners[0], corners[1], corners[2],
                            corners[3]) > 0);

  auto row = owner._corners[std::size_t(tet)];
  auto link = owner._neighbors[std::size_t(tet)];
  for (std::size_t slot = 0; slot < 4; ++slot) {
    row[slot] = corners[slot];
    link[slot] = Owner::none;
  }
  if (std::size_t(tet) < owner._protected_slots.size()) {
    owner._protected_slots[std::size_t(tet)] = 0;
    for (std::size_t slot = 0; slot < 4; ++slot)
      owner._owners[4 * std::size_t(tet) + slot] = Owner::none;
  }
  if (owner._record_site_cells && corners[3] != Owner::infinite)
    for (auto corner : corners)
      owner._site_cells.store(std::size_t(corner), tet);
  if (owner._record_written && corners[3] != Owner::infinite)
    owner._written.push_back(tet);
  return base;
}

} // namespace tf::topology::cdt::dt3
