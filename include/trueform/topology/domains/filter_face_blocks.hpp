/*
 * Copyright (c) 2025 XLAB
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
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/zip.hpp"

namespace tf::topology::domains {

/// @ingroup topology_components
/// @brief Filter each block of an offset-blocked buffer of face indices,
/// dropping elements whose mask entry is truthy.
///
/// @param input offset-blocked buffer indexed by face id.
/// @param exclude_mask per-face range; `exclude_mask[f]` truthy marks
///        `f` as excluded; matching entries are removed from every block.
/// @return A new offset-blocked buffer with the same number of blocks
///         as the input; each block is the input block with excluded
///         elements removed.
template <typename Index, typename T, typename Mask>
auto filter_face_blocks(const tf::offset_block_buffer<Index, T> &input,
                        const Mask &exclude_mask)
    -> tf::offset_block_buffer<Index, T> {
  auto n_blocks = Index(input.size());

  tf::offset_block_buffer<Index, T> out;
  auto &new_offsets = out.offsets_buffer();
  new_offsets.allocate(n_blocks + 1);
  new_offsets[0] = 0;
  tf::parallel_for_each(
      tf::zip(input, tf::drop(new_offsets, 1)),
      [&](auto pair) {
        auto &&[block, count] = pair;
        count = 0;
        for (auto v : block)
          if (!exclude_mask[v])
            ++count;
      },
      tf::checked);
  for (Index i = 0; i < n_blocks; ++i) {
    new_offsets[i + 1] += new_offsets[i];
  }

  out.data_buffer().allocate(new_offsets[n_blocks]);
  tf::parallel_for_each(
      tf::zip(input, out),
      [&](auto pair) {
        auto &&[src, dst] = pair;
        auto it = dst.begin();
        for (auto v : src)
          if (!exclude_mask[v])
            *it++ = v;
      },
      tf::checked);

  return out;
}

} // namespace tf::topology::domains
