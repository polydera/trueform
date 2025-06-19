/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./blocked_buffer.hpp"
#include "./implementation/compute_manifold_edge_link.hpp"

namespace tf {
template <typename Index, std::size_t N>
class manifold_edge_link : public blocked_buffer<manifold_edge_peer<Index>, N> {
  using base_t = blocked_buffer<manifold_edge_peer<Index>, N>;

public:
  template <typename Range>
  auto build(const Range &blocks, const tf::face_membership<Index> &blink) -> void {
    base_t::data_buffer().allocate(blocks.size() * N);
    implementation::compute_manifold_edge_link(blocks, blink, *this);
  }
};

} // namespace tf
