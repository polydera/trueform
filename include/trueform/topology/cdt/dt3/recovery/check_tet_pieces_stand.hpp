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
#include <vector>

namespace tf::topology::cdt::dt3 {

/// Whether every piece of `pieces`, sites with slots ascending, is an edge
/// of the published complex `owner`. An oracle: the phases state it
/// under verification, and tests call it.
template <typename Owner, typename Pieces>
auto check_tet_pieces_stand(const Owner &owner, const Pieces &pieces) -> bool {
  using Index = typename Owner::index_type;
  using key_t = std::array<Index, 2>;
  std::vector<key_t> edges;
  for (std::size_t c = 0; c < owner._n_finite; ++c) {
    const auto corners = owner._corners[c];
    for (std::size_t i = 0; i < 4; ++i)
      for (std::size_t j = i + 1; j < 4; ++j)
        edges.push_back({std::min(corners[i], corners[j]),
                         std::max(corners[i], corners[j])});
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  for (const auto &piece : pieces)
    if (!std::binary_search(edges.begin(), edges.end(),
                            key_t{piece[0], piece[1]}))
      return false;
  return true;
}

} // namespace tf::topology::cdt::dt3
