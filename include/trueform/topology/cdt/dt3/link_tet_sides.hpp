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
#include "../../../core/buffer.hpp"
#include "./tet_side.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

template <typename Owner, typename Index>
auto link_tet_sides(Owner &owner, tf::buffer<tet_side<Index>> &sides) -> void {
  std::sort(sides.begin(), sides.end(),
            [](const tet_side<Index> &x, const tet_side<Index> &y) {
              if constexpr (sizeof(Index) <= sizeof(std::uint32_t)) {
                const auto a = (std::uint64_t(std::uint32_t(x.a)) << 32) |
                               std::uint32_t(x.b);
                const auto b = (std::uint64_t(std::uint32_t(y.a)) << 32) |
                               std::uint32_t(y.b);
                return a < b;
              } else
                return x.a < y.a || (x.a == y.a && x.b < y.b);
            });
  assert(sides.size() % 2 == 0);
  for (std::size_t i = 0; i + 1 < sides.size(); i += 2) {
    const auto x = sides[i];
    const auto y = sides[i + 1];
    assert(x.a == y.a && x.b == y.b);
    owner._neighbors[std::size_t(x.tet)][std::size_t(x.slot)] = y.tet;
    owner._neighbors[std::size_t(y.tet)][std::size_t(y.slot)] = x.tet;
  }
}

} // namespace tf::topology::cdt::dt3
