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
#include <limits>

namespace tf::arrangement {

/// THE BLOCK RESTATE: every restatement of a tier's appended blocks is this
/// one framer — the appended offsets, the parallel per-plane write, the
/// ticket. The caller states only the block's content: `write_block(at,
/// plane, begin, local)` writes the plane's rows from `begin` and returns
/// the end; the framer states the block's offset at that end and tickets the
/// plane to its appended block. `row_prefix` is the caller's proven extent
/// per plane — `prove_plane_block_extents` below, or an earlier proof of the
/// caller's own.
template <typename Index, typename Prefix, typename WriteBlock, typename Local>
auto restate_plane_blocks(tf::buffer<Index> &block_offsets,
                          tf::buffer<Index> &block_rows,
                          tf::buffer<Index> &plane_ticket,
                          const tf::buffer<Index> &planes,
                          const Prefix &row_prefix, WriteBlock &&write_block,
                          Local local) -> void {
  if (block_offsets.size() == 0)
    block_offsets.push_back(Index(0));
  const auto block_base = Index(block_offsets.size() - 1);
  const auto row_base = std::size_t(block_rows.size());
  const auto old_block_offsets = block_offsets.size();
  block_offsets.reallocate(old_block_offsets + planes.size());
  block_rows.reallocate(row_base +
                        std::size_t(row_prefix[planes.size()]));
  tf::parallel_for_each(
      tf::make_sequence_range(planes.size()),
      [&](std::size_t at, Local &state) {
        const auto plane = planes[at];
        const auto end = write_block(
            at, plane, row_base + std::size_t(row_prefix[at]), state);
        block_offsets[old_block_offsets + at] = Index(end);
        plane_ticket[std::size_t(plane)] = block_base + Index(at);
      },
      static_cast<Local &&>(local));
}

/// The checked prefix of the restate's extents: one count per plane in
/// parallel, one serial prefix proving every partial sum, the appended
/// block and row extents proven against the index space. False means the
/// index space cannot hold the statement and nothing was touched.
template <typename Index, typename CountOf>
auto prove_plane_block_extents(std::size_t n_blocks, std::size_t n_rows,
                               const tf::buffer<Index> &planes,
                               CountOf &&count_of,
                               tf::buffer<std::size_t> &row_prefix) -> bool {
  row_prefix.allocate(planes.size() + 1);
  row_prefix[0] = 0;
  tf::parallel_for_each(
      tf::make_sequence_range(planes.size()),
      [&](std::size_t at) { row_prefix[at + 1] = count_of(planes[at]); });
  const auto index_extent = std::size_t(std::numeric_limits<Index>::max());
  for (std::size_t at = 0; at < planes.size(); ++at) {
    if (row_prefix[at + 1] > index_extent - row_prefix[at])
      return false;
    row_prefix[at + 1] += row_prefix[at];
  }
  return planes.size() <= index_extent - n_blocks &&
         row_prefix[planes.size()] <= index_extent - n_rows;
}

} // namespace tf::arrangement
