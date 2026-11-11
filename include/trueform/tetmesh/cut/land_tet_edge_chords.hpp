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
#include "./state_tet_chord_crossings.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// A chord along an edge of its face cuts nothing: the points it holds
/// between its ends (`chord_points`) land on that edge, `edge_key(t, k)` of
/// the cut triangle `cut[face]`, in `edge_points` (sorted `{a, b, point}`).
/// `run` is scratch. O(chords + chord points + edge points log edge points).
template <typename Int, typename Index, typename Cut, typename Point,
          typename EdgeKey>
auto land_tet_edge_chords(const tf::buffer<tet_cut_chord<Index>> &chords,
                          const tf::buffer<std::array<Index, 3>> &chord_points,
                          const Cut &cut, const Point &point,
                          const EdgeKey &edge_key,
                          tf::buffer<std::array<Index, 3>> &edge_points,
                          tf::buffer<Index> &run) -> void {
  for (const auto &chord : chords)
    if (chord.edge >= 0) {
      state_tet_chord_run<Int>(chord.a, chord.b, chord_points, point, run);
      const auto key =
          edge_key(cut[std::size_t(chord.face)], std::size_t(chord.edge));
      for (std::size_t k = 1; k + 1 < run.size(); ++k)
        edge_points.push_back({key[0], key[1], run[k]});
    }
  std::sort(edge_points.begin(), edge_points.end());
  edge_points.erase_till_end(
      std::unique(edge_points.begin(), edge_points.end()));
}

} // namespace tf::tetmesh
