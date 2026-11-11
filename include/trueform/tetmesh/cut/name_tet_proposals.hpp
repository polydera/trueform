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
#include "../../core/buffer.hpp"
#include "../../core/points_buffer.hpp"
#include "../../exact/vertex.hpp"
#include "./state_tet_chord_crossings.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// Name the chord crossings `proposed`, id `proposal_base + i` for proposal
/// `i`, by the identity the drain's `authority` names at each position, a
/// crossing being minted for no pair, every identity it mints appended to
/// `points` at its position, in mint order; `crossings` and `chord_points`
/// are rewritten to the names, a point naming an end of its chord dropped.
/// `named` and `no_pairs` are scratch. O(proposals log proposals + chord
/// points log chord points).
template <typename Index, typename Int, typename Authority>
auto name_tet_proposals(tf::points_buffer<Int, 3> &points,
                        const tf::buffer<tf::exact::pt3<Int>> &proposed,
                        Index proposal_base, Authority &authority,
                        tf::buffer<tet_chord_crossing<Index>> &crossings,
                        tf::buffer<std::array<Index, 3>> &chord_points,
                        tf::buffer<Index> &named,
                        tf::buffer<std::array<Index, 2>> &no_pairs) -> void {
  if (!proposed.size())
    return;
  no_pairs.allocate(proposed.size());
  std::fill(no_pairs.begin(), no_pairs.end(),
            std::array<Index, 2>{Index(-1), Index(-1)});
  const Index first = authority.name_positions(proposed, no_pairs, named);
  for (Index id = first; id < authority.n_identities(); ++id)
    points.push_back(authority.position_of(id));
  const auto final_of = [&named, proposal_base](Index p) {
    return p < proposal_base ? p : named[std::size_t(p - proposal_base)];
  };
  for (auto &crossing : crossings)
    crossing.point = final_of(crossing.point);
  std::size_t w = 0;
  for (std::size_t r = 0; r < chord_points.size(); ++r) {
    auto entry = chord_points[r];
    entry[2] = final_of(entry[2]);
    if (entry[2] != entry[0] && entry[2] != entry[1])
      chord_points[w++] = entry;
  }
  chord_points.erase_till_end(chord_points.begin() + std::ptrdiff_t(w));
  std::sort(chord_points.begin(), chord_points.end());
  chord_points.erase_till_end(
      std::unique(chord_points.begin(), chord_points.end()));
}

} // namespace tf::tetmesh
