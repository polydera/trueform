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
#include "../../../../core/config/verify.hpp"
#include "../../../../core/buffer.hpp"
#include "../../../../core/offset_block_buffer.hpp"
#include "./state_tet_site_cells.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Find the facet of the complex `owner` whose corners are the three sites
/// of `triangle`, read off the cells of whichever of them holds the fewest
/// in `site_cells`, stated for the ascending `sites`
/// (@ref state_tet_site_cells): every cell holding the
/// facet holds all three. Returns whether it stands, `ticket` then
/// `4 cell + slot` from the lower of the facet's two cells.
template <typename Owner, typename Index, typename Triangle>
auto find_tet_face_facet(
    const Owner &owner, const tf::buffer<Index> &sites,
    const tf::offset_block_buffer<Index, Index> &site_cells,
    const Triangle &triangle, Index &ticket) -> bool {
  const std::array<Index, 3> corners{Index(triangle[0]), Index(triangle[1]),
                                     Index(triangle[2])};
  const auto held_by = [&sites, &site_cells](Index site) {
    return tet_site_cells_of(sites, site_cells, site).size();
  };
  Index fewest = corners[0];
  for (std::size_t k = 1; k < 3; ++k)
    fewest = held_by(corners[k]) < held_by(fewest) ? corners[k] : fewest;
  for (auto cell : tet_site_cells_of(sites, site_cells, fewest)) {
    const auto &row = owner._corners[std::size_t(cell)];
    std::size_t held = 0, slot = 0;
    for (std::size_t k = 0; k < 4; ++k) {
      const bool on = row[k] == corners[0] || row[k] == corners[1] ||
                      row[k] == corners[2];
      held += on;
      slot = on ? slot : k;
    }
    if (held != 3)
      continue;
    const Index across = owner._neighbors[std::size_t(cell)][slot];
    if (across < cell) {
      std::size_t back = 0;
      while (back < 4 && owner._neighbors[std::size_t(across)][back] != cell)
        ++back;
      if constexpr (tf::core::verification)
        tf::core::verify(back < 4, "the mate names the cell back");
      ticket = Index(4 * std::size_t(across) + back);
    } else {
      ticket = Index(4 * std::size_t(cell) + slot);
    }
    return true;
  }
  return false;
}

} // namespace tf::topology::cdt::dt3
