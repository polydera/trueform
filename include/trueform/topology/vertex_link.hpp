/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/offset_block_buffer.hpp"
#include "./face_membership.hpp"
#include "./scoped_face_membership.hpp"
#include "./structures/compute_vertex_link.hpp"

namespace tf {
template <typename Index>
class vertex_link : public offset_block_buffer<Index, Index> {
  using base_t = offset_block_buffer<Index, Index>;

public:
  template <typename Range>
  auto build(const Range &blocks, const tf::face_membership<Index> &blink)
      -> void {
    base_t::offsets_buffer().allocate(blink.size() + 1);
    // perfect mesh has valence 6
    base_t::data_buffer().reserve(blink.size() * 4);
    topology::compute_vertex_link(blocks, blink, base_t::offsets_buffer(),
                                  base_t::data_buffer());
  }

  template <typename Range, typename SubIndex>
  auto build(const Range &blocks,
             const tf::scoped_face_membership<Index, SubIndex> &blink) -> void {
    base_t::offsets_buffer().allocate(blink.size() + 1);
    // perfect mesh has valence 6
    base_t::data_buffer().reserve(blink.size() * 4);
    topology::compute_vertex_link(blocks, blink, base_t::offsets_buffer(),
                                  base_t::data_buffer());
  }
};

} // namespace tf
