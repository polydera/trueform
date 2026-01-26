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
#include "../core/edges.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/views/zip.hpp"
#include "./edge_membership_like.hpp"

namespace tf {

/// @ingroup topology_connectivity
/// @brief Stores outgoing edges from each edge's endpoint.
///
/// For each directed edge, stores the indices of edges that start where
/// this edge ends. This enables efficient traversal along edge chains
/// and is useful for path finding algorithms.
///
/// @tparam Index The integer type for edge indices.
template <typename Index>
class directed_edge_link : public offset_block_buffer<Index, Index> {
  using base_t = offset_block_buffer<Index, Index>;

public:
  /// @brief Build directed edge links from edges and edge membership.
  /// @tparam Policy The edges policy type.
  /// @tparam Policy1 The edge membership policy type.
  /// @param edges The edges range.
  /// @param em The edge membership structure (with forward orientation).
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
