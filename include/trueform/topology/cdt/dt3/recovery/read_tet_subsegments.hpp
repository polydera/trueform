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
/// of its first. The stars are the cells of the rows' first ends (@ref
/// state_tet_site_cells), stated once for the pass into `recovery.site_cells`
/// beside the ascending ends `recovery.star_sites`; a row keeps the first cell
/// of its first end as its next hint.
template <typename DT, typename Index, typename Coord>
auto read_tet_subsegments(const DT &dt,
                          tet_segment_recovery<Index, Coord> &recovery)
    -> void {
  const std::size_t n = recovery.dirty.size();
  auto &site_cells = recovery.site_cells;
  auto &sites = recovery.star_sites;
  sites.allocate(n);
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&recovery, &sites](std::size_t i) {
        sites[i] = recovery.subsegments[std::size_t(recovery.dirty[i])].a;
      },
      tf::checked);
  state_tet_site_cells(dt, sites, site_cells, recovery.site_scratch);
  recovery.queries.allocate(n);
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&recovery, &sites, &site_cells, &dt](std::size_t i) {
        const Index k = recovery.dirty[i];
        auto &row = recovery.subsegments[std::size_t(k)];
        const auto star = tet_site_cells_of(sites, site_cells, row.a);
        read_tet_subsegment(dt, star, row,
                            star.size() ? Index(star[0]) : row.hint,
                            recovery.queries[i]);
      },
      tf::checked(64));
}

} // namespace tf::topology::cdt::dt3
