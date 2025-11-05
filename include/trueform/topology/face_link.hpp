/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/offset_block_buffer.hpp"
#include "./face_membership.hpp"
#include "./structures/compute_face_link.hpp"

namespace tf {
template <typename Index>
class face_link : public offset_block_buffer<Index, Index> {
  using base_t = offset_block_buffer<Index, Index>;

public:
  template <typename Range>
  auto build(const Range &blocks, const tf::face_membership<Index> &blink)
      -> void {
    base_t::offsets_buffer().allocate(blocks.size() + 1);
    // assume a closed triangular mesh
    base_t::data_buffer().reserve(blocks.size() * 3);
    topology::compute_face_link(blocks, blink, base_t::offsets_buffer(),
                                base_t::data_buffer());
  }
};

} // namespace tf
