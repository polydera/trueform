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
#include "../../../../core/memory.hpp"
#include "../tet_state.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Whether every piece of `pieces`, sites with slots ascending, is an edge
/// of a live finite cell of `owner`, published or as its rows stand. An
/// oracle: the phases state it under verification, and tests call it.
template <typename Owner, typename Pieces>
auto check_tet_pieces_stand(const Owner &owner, const Pieces &pieces) -> bool {
  using Index = typename Owner::index_type;
  using key_t = std::array<Index, 2>;
  tf::core::std_vector<key_t> edges;
  for (std::size_t c = 0; c < std::size_t(owner._corners.size()); ++c) {
    if (owner._states[c] != tet_state::live)
      continue;
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
