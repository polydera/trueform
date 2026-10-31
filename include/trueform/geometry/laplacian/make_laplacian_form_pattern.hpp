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
#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./laplacian_form.hpp"
#include "./laplacian_system.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::geometry {

/// The columns every free row of the form states: its ring, and the ring of
/// every row in it where the form bends.
///
/// `free_of` maps a row to its free position; a value outside
/// `[0, free_ids.size())` — `-1`, or the sentinel of the index map that
/// compacted the free rows — holds the row fixed.
///
/// Each row gathers its touched columns, sorts them and drops repeats, which
/// is a tiny sort, and the rows aggregate in order so their counts become the
/// offsets as they land. Values are left to
/// @ref tf::geometry::state_laplacian_form_values.
template <typename Index>
auto make_laplacian_form_pattern(
    const tf::geometry::laplacian_system<Index> &system,
    const tf::buffer<Index> &free_ids, const tf::buffer<Index> &free_of,
    bool bends, tf::geometry::laplacian_form<Index> &form) -> void {
  struct local_t {
    tf::buffer<Index> touched;
    tf::buffer<Index> free_counts;
    tf::buffer<Index> free_columns;
    tf::buffer<Index> fixed_counts;
    tf::buffer<Index> fixed_columns;
  };
  const auto ring = [&system](tf::buffer<Index> &touched, std::size_t row) {
    touched.push_back(Index(row));
    for (Index at = system.row_offsets[row]; at < system.row_offsets[row + 1];
         ++at)
      touched.push_back(system.columns[std::size_t(at)]);
  };
  form.free.row_offsets.clear();
  form.free.row_offsets.push_back(Index(0));
  form.free.columns.clear();
  form.fixed_offsets.clear();
  form.fixed_offsets.push_back(Index(0));
  form.fixed_columns.clear();
  tf::blocked_reduce_sequenced_aggregate(
      tf::make_sequence_range(free_ids.size()), form, local_t{},
      [&](const auto &rows, local_t &local) {
        local.free_counts.clear();
        local.free_columns.clear();
        local.fixed_counts.clear();
        local.fixed_columns.clear();
        for (std::size_t i : rows) {
          const std::size_t row = std::size_t(free_ids[i]);
          local.touched.clear();
          ring(local.touched, row);
          if (bends)
            for (Index at = system.row_offsets[row];
                 at < system.row_offsets[row + 1]; ++at)
              ring(local.touched, std::size_t(system.columns[std::size_t(at)]));
          std::sort(local.touched.begin(), local.touched.end());
          const auto unique =
              std::unique(local.touched.begin(), local.touched.end());
          const std::size_t free_first = local.free_columns.size();
          const std::size_t fixed_first = local.fixed_columns.size();
          for (auto it = local.touched.begin(); it != unique; ++it) {
            const Index position = free_of[std::size_t(*it)];
            if (std::size_t(position) >= free_ids.size())
              local.fixed_columns.push_back(*it);
            else if (std::size_t(position) <= i)
              local.free_columns.push_back(position);
          }
          std::sort(local.free_columns.begin() + std::ptrdiff_t(free_first),
                    local.free_columns.end());
          local.free_counts.push_back(
              Index(local.free_columns.size() - free_first));
          local.fixed_counts.push_back(
              Index(local.fixed_columns.size() - fixed_first));
        }
      },
      [](const local_t &local, tf::geometry::laplacian_form<Index> &out) {
        for (auto count : local.free_counts)
          out.free.row_offsets.push_back(
              Index(out.free.row_offsets.back() + count));
        for (auto count : local.fixed_counts)
          out.fixed_offsets.push_back(Index(out.fixed_offsets.back() + count));
        tf::core::append(local.free_columns, out.free.columns);
        tf::core::append(local.fixed_columns, out.fixed_columns);
      },
      tf::checked);
}

} // namespace tf::geometry
