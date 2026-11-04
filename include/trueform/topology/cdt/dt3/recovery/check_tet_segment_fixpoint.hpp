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
#include "./check_tet_pieces_stand.hpp"
#include "./tet_segment_phase.hpp"
#include "./tet_segment_status.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace tf::topology::cdt::dt3 {

/// Whether phase S's `phase` stands as the phase law says it does at its
/// fixpoint, read off the published complex `owner` alone: the lane is
/// empty, and every piece of every recovered segment, two consecutive sites
/// of its chain, is an edge of the complex. In the unconstrained complex,
/// Delaunay under the kernel's one tie rule, that is the piece's strong
/// Delaunay certificate; the complex's validity and the rule are the
/// kernel's and are not read. The law's oracle: phase S states it under
/// verification, and tests call it.
template <typename Owner>
auto check_tet_segment_fixpoint(
    const Owner &owner,
    const tet_segment_phase<typename Owner::index_type,
                            typename Owner::coord_type> &phase) -> bool {
  using Index = typename Owner::index_type;
  using key_t = std::array<Index, 2>;
  if (owner._n_protected)
    return false;
  std::vector<key_t> pieces;
  const auto &segment = phase.segment;
  for (std::size_t s = 0; s < segment.status.size(); ++s) {
    if (segment.status[s] != tet_segment_status::recovered)
      continue;
    const auto chain = segment.chains[s];
    if (chain.size() < 2)
      return false;
    for (std::size_t k = 0; k + 1 < chain.size(); ++k) {
      const Index u = chain[k].site, v = chain[k + 1].site;
      pieces.push_back({std::min(u, v), std::max(u, v)});
    }
  }
  return check_tet_pieces_stand(owner, pieces);
}

} // namespace tf::topology::cdt::dt3
