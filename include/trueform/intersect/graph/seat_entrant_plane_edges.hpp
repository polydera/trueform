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
#include "../../core/checked.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::intersect::graph {

/// The plane CSR with every entrant block moved onto the plane the entrant
/// is seated on (@ref tf::intersect::graph::seat_entrant_planes). `edges`
/// arrives with the `n_cut` cut planes' blocks followed by one block per
/// entrant, and leaves with one block per plane of the whole carrier space:
/// a cut plane's own block joined by its seated entrants' blocks, a new
/// plane's the blocks of the entrants that found it. Every block stays in
/// canonical order.
template <typename Index>
auto seat_entrant_plane_edges(
    tf::offset_block_buffer<Index, Index> &edges, Index n_cut,
    const tf::offset_block_buffer<Index, Index> &seated) -> void {
  const auto &old_offsets = edges.offsets_buffer();
  const auto &old_data = edges.data_buffer();
  const auto n_planes = Index(seated.size());
  const auto block_size = [&](Index block) {
    return old_offsets[std::size_t(block) + 1] -
           old_offsets[std::size_t(block)];
  };
  tf::buffer<Index> offsets;
  offsets.allocate(std::size_t(n_planes) + 1);
  offsets[0] = 0;
  for (Index p = 0; p < n_planes; ++p) {
    Index count = p < n_cut ? block_size(p) : Index(0);
    for (const auto e : seated[std::size_t(p)])
      count += block_size(n_cut + e);
    offsets[std::size_t(p) + 1] = offsets[std::size_t(p)] + count;
  }
  tf::buffer<Index> data;
  data.allocate(std::size_t(offsets[std::size_t(n_planes)]));
  tf::parallel_for_each(
      tf::make_sequence_range(n_planes),
      [&](Index p) {
        auto write = std::size_t(offsets[std::size_t(p)]);
        const auto take = [&](Index block) {
          for (auto i = std::size_t(old_offsets[std::size_t(block)]);
               i < std::size_t(old_offsets[std::size_t(block) + 1]); ++i)
            data[write++] = old_data[i];
        };
        if (p < n_cut)
          take(p);
        for (const auto e : seated[std::size_t(p)])
          take(n_cut + e);
        // a block taken whole is already in canonical order; only one
        // joined from several is sorted
        const auto sources =
            std::size_t(p < n_cut) + seated[std::size_t(p)].size();
        if (sources > 1)
          std::sort(data.begin() + std::ptrdiff_t(offsets[std::size_t(p)]),
                    data.begin() + std::ptrdiff_t(write));
      },
      tf::checked);
  edges.offsets_buffer() = std::move(offsets);
  edges.data_buffer() = std::move(data);
}

} // namespace tf::intersect::graph
