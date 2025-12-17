/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/edges.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/views/zip.hpp"
#include "./edge_membership_like.hpp"

namespace tf {
template <typename Index>
class directed_edge_link : public offset_block_buffer<Index, Index> {
  using base_t = offset_block_buffer<Index, Index>;

public:
  template <typename Policy, typename Policy1>
  auto build(const tf::edges<Policy> &edges,
             const tf::edge_membership_like<Policy1> &em) {
    base_t::offsets_buffer().allocate(edges.size() + 1);
    for (auto &&[o, edge] :
         tf::zip(tf::make_range(base_t::offsets_buffer().begin(), edges.size()),
                 edges))
      o = em[edge[1]].size();
    base_t::offsets_buffer().back() = 0;
    for (Index i = 1; i < Index(edges.size()) + 1; ++i)
      base_t::offsets_buffer()[i] += base_t::offsets_buffer()[i - 1];
    base_t::data_buffer().allocate(base_t::offsets_buffer().back());
    for (auto &&[o, edge] :
         tf::zip(tf::make_range(base_t::offsets_buffer().begin(), edges.size()),
                 edges))
      for (auto edge_id : em[edge[1]])
        base_t::data_buffer()[--o] = edge_id;
  }
};

} // namespace tf
