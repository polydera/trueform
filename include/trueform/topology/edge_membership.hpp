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
#include "../core/segments.hpp"
#include "./edge_membership_like.hpp"
#include "./edge_orientation.hpp"
#include "./structures/compute_face_membership.hpp"

namespace tf {

/// @ingroup topology_connectivity
/// @brief Maps each vertex to the edges incident on it.
///
/// For each vertex index, stores the list of edge indices that include
/// that vertex. Supports different edge orientations:
/// - `forward`: Only count edges where this vertex is the first endpoint
/// - `reverse`: Only count edges where this vertex is the second endpoint
/// - `bidirectional`: Count edges where this vertex is either endpoint
///
/// @tparam Index The integer type for vertex and edge indices.
template <typename Index>
class edge_membership
    : public edge_membership_like<offset_block_buffer<Index, Index>> {
  using base_t = edge_membership_like<offset_block_buffer<Index, Index>>;

public:
  /// @brief Build from edges with specified orientation.
  /// @tparam Policy The edges policy type.
  /// @param edges The edges range.
  /// @param n_unique_ids The number of unique vertex ids.
  /// @param eo The edge orientation mode (default: bidirectional).
  template <typename Policy>
  auto build(const tf::edges<Policy> &edges, std::size_t n_unique_ids,
             tf::edge_orientation eo = tf::edge_orientation::bidirectional)
      -> void {
    base_t::offsets_buffer().allocate(n_unique_ids + 1);
    switch (eo) {
    case edge_orientation::forward:
      base_t::data_buffer().allocate(edges.size());
      return build_forward(edges);
    case edge_orientation::reverse:
      base_t::data_buffer().allocate(edges.size());
      return build_reverse(edges);
    case edge_orientation::bidirectional:
      base_t::data_buffer().allocate(edges.size() * 2);
      return build_bidirectional(edges);
    }
  }

  /// @brief Build from segments with specified orientation.
  /// @tparam Policy The segments policy type.
  /// @param segments The segments range.
  /// @param eo The edge orientation mode (default: bidirectional).
  template <typename Policy>
  auto build(const segments<Policy> &segments,
             tf::edge_orientation eo = tf::edge_orientation::bidirectional)
      -> void {
    auto n_unique_ids = segments.points().size();
    build(segments.edges(), n_unique_ids, eo);
  }

private:
  template <typename Range> auto build_forward(const Range &edges) -> void {
    topology::compute_face_membership(
        tf::make_mapped_range(
            edges,
            [](const auto &r) { return std::array<Index, 1>{Index(r[0])}; }),
        base_t::offsets_buffer(), base_t::data_buffer());
  }

  template <typename Range> auto build_reverse(const Range &edges) -> void {
    topology::compute_face_membership(
        tf::make_mapped_range(
            edges,
            [](const auto &r) { return std::array<Index, 1>{Index(r[1])}; }),
        base_t::offsets_buffer(), base_t::data_buffer());
  }

  template <typename Range>
  auto build_bidirectional(const Range &edges) -> void {
    topology::compute_face_membership(edges, base_t::offsets_buffer(),
                                      base_t::data_buffer());
  }
};

} // namespace tf
