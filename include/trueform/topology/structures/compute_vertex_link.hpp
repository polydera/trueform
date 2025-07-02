/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/algorithm/circular_decrement.hpp"
#include "../../core/algorithm/circular_increment.hpp"
#include "../../core/algorithm/generate_offset_blocks.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../face_membership.hpp"
#include "../scoped_face_membership.hpp"

namespace tf::topology {
template <typename Range, typename Index>
auto compute_vertex_link(const Range &input_blocks,
                         const tf::face_membership<Index> &blink,
                         tf::buffer<Index> &offsets, tf::buffer<Index> &data) {
  auto fill_f = [&input_blocks, &blink](Index id, tf::buffer<Index> &buff) {
    auto old_size = buff.size();
    for (const auto &block : tf::make_indirect_range(blink[id], input_blocks)) {
      Index sub_id = std::find(block.begin(), block.end(), id) - block.begin();
      auto push_f = [&](auto n_id) {
        if (std::find(buff.begin() + old_size, buff.end(), block[n_id]) ==
            buff.end())
          buff.push_back(block[n_id]);
      };
      Index size = block.size();
      push_f(tf::circular_decrement(sub_id, size));
      push_f(tf::circular_increment(sub_id, size));
    }
  };
  tf::generate_offset_blocks(tf::make_sequence_range(blink.size()), offsets,
                             data, fill_f);
}

template <typename Range, typename Index>
auto compute_vertex_link(const Range &input_blocks,
                         const tf::face_membership<Index> &blink,
                         tf::offset_block_buffer<Index, Index> &buff) {
  compute_vertex_link(input_blocks, blink, buff.offsets_buffer(),
                      buff.data_buffer());
}

template <typename Range, typename Index, typename SubIndex>
auto compute_vertex_link(
    const Range &input_blocks,
    const tf::scoped_face_membership<Index, SubIndex> &blink,
    tf::buffer<Index> &offsets, tf::buffer<Index> &data) {
  auto fill_f = [&input_blocks, &blink](Index id, tf::buffer<Index> &buff) {
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
