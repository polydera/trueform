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
#include "../../../exact/orient3d.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Write corners and reset neighbors, preserving orientation while moving the
/// infinity corner to slot three. Returns the final slot of c3.
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
         tf::exact::orient3d_value_scaled<typename Owner::int_type>(
             owner._sites[std::size_t(corners[0])].pt,
             owner._sites[std::size_t(corners[1])].pt,
             owner._sites[std::size_t(corners[2])].pt,
             owner._sites[std::size_t(corners[3])].pt) > 0);

  auto row = owner._corners[std::size_t(tet)];
  auto link = owner._neighbors[std::size_t(tet)];
  for (std::size_t slot = 0; slot < 4; ++slot) {
    row[slot] = corners[slot];
    link[slot] = Owner::none;
  }
  return base;
}

} // namespace tf::topology::cdt::dt3
