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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_tables.hpp"

#include <cstddef>

namespace tf::arrangement {

/// THE GROUP MINT: every incremental append of groups to a tier's
/// definitions is this one framer — the span reallocation and the group
/// offsets, which state the canonical extent. The caller states only each
/// item's content: `write_group(item, begin)` writes item `item`'s instances
/// from row `begin` and returns the rank, among the `n_mints` this call mints,
/// of the group they are, or `-1` for an item that mints none; the framer
/// closes that group's span at `row_prefix[item + 1]` past the rows the
/// tier held. `row_prefix` is the caller's proven extent per item.
template <typename Index, typename Int, typename WriteGroup>
auto mint_plane_groups(tf::intersect::graph::plane_tables<Index, Int> &tables,
                       std::size_t n_items, std::size_t n_mints,
                       const tf::buffer<Index> &row_prefix,
                       const WriteGroup &write_group) -> void {
  auto &defs = tables.defs();
  auto &def_offsets = tables.def_offsets();
  const auto row_base = Index(defs.size());
  defs.reallocate(std::size_t(row_base) + std::size_t(row_prefix[n_items]));
  if (n_mints != 0 && def_offsets.size() == 0)
    def_offsets.push_back(Index(0));
  const auto old_def_offsets = def_offsets.size();
  def_offsets.reallocate(old_def_offsets + n_mints);
  tf::parallel_for_each(
      tf::make_sequence_range(n_items), [&](std::size_t item) {
        const auto rank =
            write_group(item, std::size_t(row_base + row_prefix[item]));
        if (rank != Index(-1))
          def_offsets[old_def_offsets + std::size_t(rank)] =
              row_base + row_prefix[item + 1];
      });
}

} // namespace tf::arrangement
