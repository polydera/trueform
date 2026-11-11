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
#include "./tet_segment_phase.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The sites of a face's rim as phase S states them: `f(face, emit)` emits
/// both ends of every row of every segment the face's rim runs along
/// (@ref tet_segment_phase `edges`). O(the rim's rows).
template <typename Index, typename Coord>
auto tet_face_rim_sites(const tet_segment_phase<Index, Coord> &phase) {
  return [&phase](Index face, const auto &emit) {
    const auto &segment = phase.segment;
    const auto &rows = segment.subsegments;
    for (const auto s :
         phase.edges[std::size_t(phase.edge_block[std::size_t(face)])])
      for (Index k = segment.first[std::size_t(s)]; k != Index(-1);
           k = rows[std::size_t(k)].next) {
        emit(rows[std::size_t(k)].a);
        emit(rows[std::size_t(k)].b);
      }
  };
}

} // namespace tf::topology::cdt::dt3
