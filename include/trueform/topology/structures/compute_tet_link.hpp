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
#include "../../core/algorithm/generate_offset_blocks.hpp"
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../tet_facet_neighbors.hpp"
#include "../tet_membership_like.hpp"
#include <iterator>

namespace tf::topology {
template <typename Range, typename Policy, typename Index>
auto compute_tet_link(const Range &tetras,
                      const tf::tet_membership_like<Policy> &blink,
                      tf::buffer<Index> &offsets, tf::buffer<Index> &data) {

  auto fill_f = [&](Index tet_id, tf::buffer<Index> &ids) {
    const auto &tet = tetras[tet_id];
    for (Index slot = 0; slot < 4; ++slot)
      tf::tet_facet_neighbors(
          blink, tet_id, Index(tet[(slot + 1) & 3]), Index(tet[(slot + 2) & 3]),
          Index(tet[(slot + 3) & 3]), std::back_inserter(ids));
  };
  tf::generate_offset_blocks(tf::make_sequence_range(tetras.size()), offsets,
                             data, fill_f);
}

template <typename Range, typename Policy, typename Index>
auto compute_tet_link(const Range &tetras,
                      const tf::tet_membership_like<Policy> &blink,
                      tf::offset_block_buffer<Index, Index> &buff) {
  compute_tet_link(tetras, blink, buff.offsets_buffer(), buff.data_buffer());
}

} // namespace tf::topology
