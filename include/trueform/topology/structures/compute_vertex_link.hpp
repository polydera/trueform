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
#include "../../core/algorithm/circular_decrement.hpp"
#include "../../core/algorithm/circular_increment.hpp"
#include "../../core/algorithm/generate_offset_blocks.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../face_membership_like.hpp"
#include "../scoped_face_membership.hpp"

namespace tf::topology {
template <typename Range, typename Policy, typename Index>
auto compute_vertex_link(const Range &input_blocks,
                         const tf::face_membership_like<Policy> &blink,
                         tf::buffer<Index> &offsets, tf::buffer<Index> &data) {
  auto fill_f = [&input_blocks, &blink](auto id, tf::buffer<Index> &buff) {
    auto old_size = buff.size();
    for (const auto &block : tf::make_indirect_range(blink[id], input_blocks)) {
      Index sub_id = static_cast<Index>(
          std::find(block.begin(), block.end(), static_cast<Index>(id)) -
          block.begin());
      auto push_f = [&](auto n_id) {
        if (std::find(buff.begin() + old_size, buff.end(), block[n_id]) ==
            buff.end())
          buff.push_back(block[n_id]);
      };
      Index size = static_cast<Index>(block.size());
      push_f(tf::circular_decrement(sub_id, size));
      push_f(tf::circular_increment(sub_id, size));
    }
  };
  tf::generate_offset_blocks(tf::make_sequence_range(blink.size()), offsets,
                             data, fill_f);
}

template <typename Range, typename Policy, typename Index>
auto compute_vertex_link(const Range &input_blocks,
                         const tf::face_membership_like<Policy> &blink,
                         tf::offset_block_buffer<Index, Index> &buff) {
  compute_vertex_link(input_blocks, blink, buff.offsets_buffer(),
                      buff.data_buffer());
}

template <typename Range, typename Index, typename SubIndex>
auto compute_vertex_link(
    const Range &input_blocks,
    const tf::scoped_face_membership<Index, SubIndex> &blink,
    tf::buffer<Index> &offsets, tf::buffer<Index> &data) {
  auto fill_f = [&input_blocks, &blink](auto id, tf::buffer<Index> &buff) {
    auto old_size = buff.size();
    for (const auto &[block_id, sub_id] : blink[id]) {
      const auto &block = input_blocks[block_id];
      auto push_f = [&](auto n_id) {
        if (std::find(buff.begin() + old_size, buff.end(), block[n_id]) ==
            buff.end())
          buff.push_back(block[n_id]);
      };
      Index size = block.size();
      push_f(tf::circular_decrement(Index(sub_id), size));
      push_f(tf::circular_increment(Index(sub_id), size));
    }
  };
  tf::generate_offset_blocks(tf::make_sequence_range(blink.size()), offsets,
                             data, fill_f);
}

template <typename Range, typename Index, typename SubIndex>
auto compute_vertex_link(
    const Range &input_blocks,
    const tf::scoped_face_membership<Index, SubIndex> &blink,
    tf::offset_block_buffer<Index, Index> &buff) {
  compute_vertex_link(input_blocks, blink, buff.offsets_buffer(),
                      buff.data_buffer());
}
} // namespace tf::topology
