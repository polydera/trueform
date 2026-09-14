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

#include "./chart_cell.hpp"
#include "./exact_dot.hpp"
#include "./exact_lane.hpp"
#include "./line_cells.hpp"
#include "./pool_records.hpp"

#include "../../../core/algorithm/generic_generate.hpp"
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"

#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#ifdef TF_POOL_CENSUS
#include <chrono>
#endif

namespace tf::exact::door::pool {

/// One compact record per name: the cell it falls in and the name itself.
/// The sorted grouping is one sort of these and one adjacent sweep.
struct line_cell_record {
  chart_cell cell{};
  int name = -1;

  auto operator<(const line_cell_record &o) const -> bool {
    if (cell != o.cell)
      return cell < o.cell;
    return name < o.name;
  }
};

/// The cells, their elections and the ticket back.
///
/// THE GROUPING IS COUNTS AND ONE PREFIX. The chart is `3 (2K + 1)^2`
/// positions and a key IS its own position in that domain
/// (@ref tf::exact::door::pool::chart_index_of), so the names are counted
/// into it, the counts turn into offsets in one walk, and each name is
/// written once into the slot its cell already owns. Walking the domain in
/// index order is walking the keys in canonical order. Past the size where
/// that domain is proportionate to the names standing in it, sorting the
/// records states the SAME structure — the choice moves the price and never
/// the answer.
///
/// The members of a cell come out ASCENDING BY NAME, and a name id is its
/// position in canonical exact order, so the elected least plane is the
/// first member and the election costs nothing beyond the walk that already
/// found the cell's start.
///
/// `cell_straddle_exposure` is counted here and nowhere else. Its scan is an
/// INSTRUMENT — no placement reads it — so it runs only under
/// `TF_POOL_CENSUS` and the door pays nothing for it.
template <typename Int>
auto make_line_cells(const pool_names<Int> &names, int resolution,
                     election_census &census, line_cells<Int> &cells) -> void {
  using coefficient_type = typename exact_lane<Int>::coefficient_type;

  const auto count = names.plane.size();
  cells.key.clear();
  cells.offsets.clear();
  cells.member.clear();
  cells.elected.clear();
  cells.line.clear();
  cells.square_length.clear();
  cells.support.clear();
  cells.cell_of_name.allocate(count);
  cells.sign_of_name.allocate(count);
  if (count == 0) {
    cells.offsets.push_back(0);
    return;
  }

  tf::buffer<chart_cell> key;
  key.allocate(count);
  tf::parallel_for_each(
      tf::make_sequence_range(count),
      [&names, &cells, &key, resolution](std::size_t n) {
        const auto &plane = names.plane[n];
        const std::array<coefficient_type, 3> normal{plane[0], plane[1],
                                                     plane[2]};
        key[n] = chart_cell_of<Int>(normal, resolution);
        cells.sign_of_name[n] = key[n].sign;
        cells.cell_of_name[n] = -1;
      },
      tf::checked);

  // The dense chart is a buffer indexed by the key's own position, so it is
  // worth allocating only while it is proportionate to the names standing in
  // it AND while that position is a number the key's index type states. The
  // domain is quadratic in the resolution, so this is the SMALL-CHART regime:
  // `3 (2K + 1)^2 <= n` admits `K` up to about `sqrt(n / 3) / 2` — the floor
  // of four at any scene, a scene-independent `TF_POOL_DISCOVERY_K`, and a
  // band fine enough that its renormalized pitch is a handful of steps.
  // Everything coarser sorts, and both paths state the same cells in the same
  // order with the same members ascending by name.
  const std::int64_t domain = chart_domain(resolution);
  if (domain > 0 && domain <= std::int64_t(count) &&
      domain <= std::int64_t(std::numeric_limits<int>::max())) {
    tf::buffer<int> at;
    at.allocate(std::size_t(domain));
    for (auto &held : at)
      held = 0;
    for (std::size_t n = 0; n < count; ++n)
      if (key[n].sign != 0)
        ++at[std::size_t(chart_index_of(key[n], resolution))];

    const int span = 2 * resolution + 1;
    int total = 0;
    for (int i = 0; i < int(domain); ++i) {
      const int held = at[std::size_t(i)];
      at[std::size_t(i)] = total;
      if (held == 0)
        continue;
      chart_cell around;
      around.at = {i / (span * span), (i / span) % span - resolution,
                   i % span - resolution};
      cells.key.push_back(around);
      cells.offsets.push_back(total);
      total += held;
    }
    cells.offsets.push_back(total);
    cells.member.allocate(std::size_t(total));
    for (std::size_t n = 0; n < count; ++n)
      if (key[n].sign != 0)
        cells.member[std::size_t(
            at[std::size_t(chart_index_of(key[n], resolution))]++)] = int(n);
  } else {
    // the sort's key is total — a cell then a name, and a name occurs once —
    // so the records may be gathered in any order
    tf::buffer<line_cell_record> record;
    tf::generic_generate(
        tf::make_sequence_range(count), record,
        [&key](std::size_t n, tf::buffer<line_cell_record> &out) {
          if (key[n].sign != 0)
            out.push_back(line_cell_record{key[n], int(n)});
        },
        tf::checked);
    tbb::parallel_sort(record.begin(), record.end());

    cells.member.allocate(record.size());
    for (std::size_t k = 0; k < record.size(); ++k) {
      if (k == 0 || record[k].cell != record[k - 1].cell) {
        // a cell's key is its position and nothing else; the alignment a
        // name's own key was read under is `sign_of_name`
        chart_cell around;
        around.at = record[k].cell.at;
        cells.key.push_back(around);
        cells.offsets.push_back(int(k));
      }
      cells.member[k] = record[k].name;
    }
    cells.offsets.push_back(int(record.size()));
  }

  // THE CELL IS THE GRAIN of its own election and of the ticket back: a name
  // belongs to exactly one cell, so every store below is disjoint. The block
  // count is not a proxy for the work, since one cell may hold a whole wall
  // family, so the range is not checked.
  const auto n_cells = cells.key.size();
  cells.elected.allocate(n_cells);
  cells.line.allocate(n_cells);
  cells.square_length.allocate(n_cells);
  cells.support.allocate(n_cells);
  tf::parallel_for_each(
      tf::make_sequence_range(n_cells), [&names, &cells](std::size_t c) {
        const int elected = cells.member[std::size_t(cells.offsets[c])];
        const auto &plane = names.plane[std::size_t(elected)];
        const auto sign =
            coefficient_type(cells.sign_of_name[std::size_t(elected)]);
        const std::array<coefficient_type, 3> line{
            sign * plane[0], sign * plane[1], sign * plane[2]};
        cells.elected[c] = elected;
        cells.line[c] = line;
        cells.square_length[c] = exact_dot<Int>(line, line);
        cells.support[c] = names.support_point[std::size_t(elected)];
        for (int k = cells.offsets[c]; k < cells.offsets[c + 1]; ++k)
          cells.cell_of_name[std::size_t(cells.member[std::size_t(k)])] =
              int(c);
      });
  census.cells = std::int64_t(n_cells);

#ifdef TF_POOL_CENSUS
  const auto straddle = std::chrono::steady_clock::now();
  for (std::size_t c = 0; c < cells.key.size(); ++c) {
    bool crowded = false;
    for (int db = -1; db <= 1 && !crowded; ++db)
      for (int dc = -1; dc <= 1 && !crowded; ++dc) {
        if (db == 0 && dc == 0)
          continue;
        chart_cell around;
        around.at = {cells.key[c].at[0], cells.key[c].at[1] + db,
                     cells.key[c].at[2] + dc};
        const auto found =
            std::lower_bound(cells.key.begin(), cells.key.end(), around);
        crowded = found != cells.key.end() && *found == around;
      }
    if (crowded)
      census.cell_straddle_exposure +=
          std::int64_t(cells.offsets[c + 1] - cells.offsets[c]);
  }
  census.straddle_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - straddle)
          .count();
#endif
}

} // namespace tf::exact::door::pool
