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

#include <cstddef>

namespace tf::intersect::graph {

/// THE BLOCK REBUILD: every whole restatement of a block CSR is this one
/// framer — one count per block in parallel, one serial prefix, one exact
/// allocation, one write per block in parallel. The caller states only each
/// block's content: `count_of(block)` its row count, `write_block(block,
/// begin)` its rows from `begin`. `offsets` and `rows` leave stating
/// `n_blocks` blocks; `checked` is forwarded to both parallel passes.
template <typename Index, typename CountOf, typename WriteBlock,
          typename... Checked>
auto rebuild_plane_blocks(std::size_t n_blocks, const CountOf &count_of,
                          const WriteBlock &write_block,
                          tf::buffer<Index> &offsets, tf::buffer<Index> &rows,
                          Checked... checked) -> void {
  offsets.allocate(n_blocks + 1);
  offsets[0] = Index(0);
  tf::parallel_for_each(
      tf::make_sequence_range(n_blocks),
      [&](std::size_t block) { offsets[block + 1] = count_of(block); },
      checked...);
  for (std::size_t block = 1; block <= n_blocks; ++block)
    offsets[block] += offsets[block - 1];
  rows.allocate(std::size_t(offsets[n_blocks]));
  tf::parallel_for_each(
      tf::make_sequence_range(n_blocks),
      [&](std::size_t block) {
        write_block(block, std::size_t(offsets[block]));
      },
      checked...);
}

} // namespace tf::intersect::graph
