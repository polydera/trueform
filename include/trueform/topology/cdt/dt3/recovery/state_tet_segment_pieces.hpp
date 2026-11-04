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
#include "../../../../core/algorithm/sequenced_generate.hpp"
#include "../../../../core/checked.hpp"
#include "./tet_segment_recovery.hpp"
#include "./tet_segment_status.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <tbb/parallel_sort.h>

namespace tf::topology::cdt::dt3 {

/// State into `pieces` every piece of every segment `recovery` recovered,
/// a row of it, slots ascending, sorted and once.
template <typename Index, typename Coord, typename Pieces>
auto state_tet_segment_pieces(
    const tet_segment_recovery<Index, Coord> &recovery, Pieces &pieces)
    -> void {
  const auto &rows = recovery.subsegments;
  const auto &status = recovery.status;
  pieces.clear();
  tf::sequenced_generate(
      rows, pieces,
      [&status](const auto &row, auto &out) {
        if (status[std::size_t(row.segment)] != tet_segment_status::recovered)
          return;
        out.push_back(std::array<Index, 2>{std::min(row.a, row.b),
                                           std::max(row.a, row.b)});
      },
      tf::checked);
  tbb::parallel_sort(pieces.begin(), pieces.end());
  pieces.erase_till_end(std::unique(pieces.begin(), pieces.end()));
}

} // namespace tf::topology::cdt::dt3
