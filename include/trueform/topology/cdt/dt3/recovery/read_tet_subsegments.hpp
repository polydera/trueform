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
#include "../../../../core/algorithm/parallel_for_each.hpp"
#include "../../../../core/checked.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include "./read_tet_subsegment.hpp"
#include "./state_tet_site_cells.hpp"
#include "./tet_segment_recovery.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Read every due row against the published complex into `recovery.queries`,
/// one per `recovery.dirty` row: present when its second end is in the star
/// of its first and it is not redirected, which reads it encroached. The
/// stars are the cells of every site (@ref state_tet_site_cells), stated
/// once for the pass into `recovery.site_cells`; a row keeps the first cell
/// of its first end as its next hint.
template <typename DT, typename Index, typename Coord>
auto read_tet_subsegments(const DT &dt,
                          tet_segment_recovery<Index, Coord> &recovery)
    -> void {
  const std::size_t n = recovery.dirty.size();
  auto &site_cells = recovery.site_cells;
  state_tet_site_cells(dt, site_cells);
  recovery.queries.allocate(n);
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&recovery, &site_cells, &dt](std::size_t i) {
        const Index k = recovery.dirty[i];
        auto &row = recovery.subsegments[std::size_t(k)];
        const auto star = site_cells[std::size_t(row.a)];
        read_tet_subsegment(dt, star, row, k,
                            star.size() ? Index(star[0]) : row.hint,
                            recovery.redirected, recovery.queries[i]);
      },
      tf::checked(64));
}

} // namespace tf::topology::cdt::dt3
