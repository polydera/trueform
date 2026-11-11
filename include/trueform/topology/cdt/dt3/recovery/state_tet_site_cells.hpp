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
#include "../../../../core/algorithm/generate_offset_blocks.hpp"
#include "../../../../core/algorithm/parallel_iota.hpp"
#include "../../../../core/buffer.hpp"
#include "../../../../core/checked.hpp"
#include "../../../../core/offset_block_buffer.hpp"
#include "../tet_finite_cell.hpp"
#include "./enter_tet_site_star.hpp"
#include "./gather_tet_star.hpp"
#include "./open_tet_star_stamps.hpp"
#include "./scatter_tet_records_by_key.hpp"
#include "./tet_site_cell_scratch.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The rows per asked site below which @ref state_tet_site_cells counts: a
/// star gathered costs about 1 us, a row counted about 4 ns (t1004824,
/// t1007406, one worker), so the two shapes meet near 256 rows per site.
inline constexpr std::size_t tet_site_star_rows = 256;

/// State the cells of the sites `sites` names, in any order and any number
/// of times: `sites` becomes ascending sites, each once, holding every site
/// it named, and block `i` of `site_cells` is the finite live cells of `dt`
/// holding `sites[i]`, ascending. One table, two work shapes by its density:
/// sparse sites against the rows are sorted and gather their stars from the
/// cell `dt` keeps for each across the facets holding it
/// (@ref gather_tet_star), costing the stars; dense sites become every site,
/// whose cells are grouped by site in row order
/// (@ref scatter_tet_records_by_key), costing the rows plus the sites.
/// `scratch` is the caller's.
template <typename DT, typename Index>
auto state_tet_site_cells(const DT &dt, tf::buffer<Index> &sites,
                          tf::offset_block_buffer<Index, Index> &site_cells,
                          tet_site_cell_scratch<Index> &scratch) -> void {
  const std::size_t n_rows = dt.n_tets();
  site_cells.data_buffer().clear();
  if (sites.size() * tet_site_star_rows < n_rows) {
    std::sort(sites.begin(), sites.end());
    sites.erase_till_end(std::unique(sites.begin(), sites.end()));
    auto &stamps = scratch.stamps;
    open_tet_star_stamps(dt, stamps);
    const auto entry = dt.entry();
    tf::generate_offset_blocks(
        sites, site_cells,
        [&dt, &stamps, entry](Index site, auto &star) {
          const std::size_t from = star.size();
          bool ticketed = false;
          gather_tet_star(dt, site,
                          enter_tet_site_star(dt, site, entry, ticketed),
                          stamps, star);
          std::sort(star.begin() + std::ptrdiff_t(from), star.end());
        },
        tf::checked);
    return;
  }
  const auto cells = dt.tets();
  const std::size_t n_sites = dt.n_sites();
  sites.allocate(n_sites);
  tf::parallel_iota(sites, Index(0));
  using pair_t = std::array<Index, 2>;
  scatter_tet_records_by_key(
      n_rows, n_sites,
      [&dt, &cells](std::size_t row, const auto &emit) {
        if (!tet_finite_cell(dt, Index(row)))
          return;
        for (auto corner : cells[row])
          emit(pair_t{Index(corner), Index(row)});
      },
      [](const pair_t &pair) { return pair[0]; },
      [](const pair_t &pair) { return pair[1]; }, site_cells.offsets_buffer(),
      scratch.staged, scratch.counts, site_cells.data_buffer());
}

/// The cells of `site`, one of the ascending `sites` @ref
/// state_tet_site_cells stated `site_cells` for.
template <typename Index>
auto tet_site_cells_of(const tf::buffer<Index> &sites,
                       const tf::offset_block_buffer<Index, Index> &site_cells,
                       Index site) {
  return site_cells[std::size_t(
      std::lower_bound(sites.begin(), sites.end(), site) - sites.begin())];
}

} // namespace tf::topology::cdt::dt3
