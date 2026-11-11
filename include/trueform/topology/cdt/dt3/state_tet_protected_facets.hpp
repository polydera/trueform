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
#include "./tet_facet_is_protected.hpp"
#include "./tet_facet_key.hpp"
#include "./tet_state.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// State every protected facet of `owner` in `facets`, corner slots
/// ascending and sorted, once each: the bits of the live cells and the
/// unheld.
template <typename Owner, typename Facets>
auto state_tet_protected_facets(const Owner &owner, Facets &facets) -> void {
  using Index = typename Owner::index_type;
  facets.clear();
  const std::size_t rows = std::min(std::size_t(owner._corners.size()),
                                    owner._protected_slots.size());
  for (std::size_t cell = 0; cell < rows; ++cell) {
    if (owner._states[cell] == tet_state::dead || !owner._protected_slots[cell])
      continue;
    for (std::size_t slot = 0; slot < 4; ++slot)
      if (tet_facet_is_protected(owner, Index(cell), slot)) {
        auto key = tet_facet_key(owner._corners[cell], slot);
        facets.push_back(std::array<Index, 3>{key[0], key[1], key[2]});
      }
  }
  for (const auto &record : owner._protected_unheld)
    facets.push_back(record.key);
  std::sort(facets.begin(), facets.end());
  facets.erase(std::unique(facets.begin(), facets.end()), facets.end());
}

} // namespace tf::topology::cdt::dt3
