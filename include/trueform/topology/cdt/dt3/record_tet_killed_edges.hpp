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
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Record in `killed` the edges of the cells `cavity` names, each its two
/// corner slots ascending, the hull corner's left out: the edges an
/// insertion's cavity takes, which no reader of the complex can find again
/// once the star replaces them.
template <typename Owner, typename Cavity, typename Killed>
auto record_tet_killed_edges(const Owner &owner, const Cavity &cavity,
                             Killed &killed) -> void {
  using Index = typename Owner::index_type;
  for (auto cell : cavity) {
    const auto corners = owner._corners[std::size_t(cell)];
    for (std::size_t i = 0; i < 4; ++i)
      for (std::size_t j = i + 1; j < 4; ++j)
        if (corners[i] != Owner::infinite && corners[j] != Owner::infinite)
          killed.push_back(
              std::array<Index, 2>{std::min(corners[i], corners[j]),
                                   std::max(corners[i], corners[j])});
  }
}

} // namespace tf::topology::cdt::dt3
