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
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/edges.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/polygons.hpp"
#include "../core/segments.hpp"
#include "../core/views/enumerate.hpp"
#include "./edge_membership.hpp"
#include "./face_membership_like.hpp"
#include "./scoped_face_membership.hpp"
#include "./structures/compute_vertex_link.hpp"
#include "./vertex_link_like.hpp"

namespace tf {

/// @ingroup topology_connectivity
/// @brief Stores the 1-ring neighborhood for each vertex.
///
/// For each vertex, stores the indices of neighboring vertices (or faces,
/// depending on how it was built). This is essential for mesh operations
/// like smoothing, boundary detection, and local neighborhood queries.
///
/// Build methods:
/// - From @ref tf::faces and @ref tf::face_membership_like
/// - From @ref tf::polygons and @ref tf::face_membership_like
/// - From @ref tf::polygons and @ref tf::scoped_face_membership
/// - From @ref tf::edges or @ref tf::segments with edge orientation
///
/// @tparam Index The integer type for vertex indices.
template <typename Index>
class vertex_link : public vertex_link_like<offset_block_buffer<Index, Index>> {
  using b_base_t = offset_block_buffer<Index, Index>;
  using base_t = vertex_link_like<b_base_t>;

public:
  /// @brief Build from faces and face membership.
  /// @tparam Policy0 The faces policy type.
  /// @tparam Policy1 The face membership policy type.
  /// @param faces The faces range.
  /// @param blink The face membership structure.
  template <typename Policy0, typename Policy1>
  auto build(const tf::faces<Policy0> &faces,
             const tf::face_membership_like<Policy1> &blink) -> void {
    base_t::offsets_buffer().allocate(blink.size() + 1);
    // perfect mesh has valence 6
    base_t::data_buffer().reserve(blink.size() * 4);
    topology::compute_vertex_link(faces, blink, base_t::offsets_buffer(),
                                  base_t::data_buffer());
  }

  /// @brief Build from polygons and face membership.
  /// @tparam Policy0 The polygons policy type.
  /// @tparam Policy1 The face membership policy type.
  /// @param polygons The polygons range.
  /// @param blink The face membership structure.
  template <typename Policy0, typename Policy1>
  auto build(const tf::polygons<Policy0> &polygons,
             const tf::face_membership_like<Policy1> &blink) -> void {
    return build(polygons.faces(), blink);
  }

  /// @brief Build from polygons and scoped face membership.
  /// @tparam Policy The polygons policy type.
  /// @tparam SubIndex The sub-index type for scoped membership.
  /// @param polygons The polygons range.
  /// @param blink The scoped face membership structure.
  template <typename Policy, typename SubIndex>
  auto build(const tf::polygons<Policy> &polygons,
             const tf::scoped_face_membership<Index, SubIndex> &blink) -> void {
    base_t::offsets_buffer().allocate(blink.size() + 1);
    // perfect mesh has valence 6
    base_t::data_buffer().reserve(blink.size() * 4);
    topology::compute_vertex_link(polygons.faces(), blink,
                                  base_t::offsets_buffer(),
                                  base_t::data_buffer());
  }

  /// @brief Build from edges with specified orientation.
  ///
  /// Builds vertex connectivity from edge data. The orientation determines
  /// which endpoint is used: forward uses first vertex, reverse uses second,
  /// bidirectional uses both.
  ///
  /// @tparam Policy The edges policy type.
  /// @param edges The edges range.
  /// @param n_unique_ids The number of unique vertex ids.
  /// @param eo The edge orientation mode (default: bidirectional).
  template <typename Policy>
  auto build(const tf::edges<Policy> &edges, std::size_t n_unique_ids,
             tf::edge_orientation eo = tf::edge_orientation::bidirectional) {
    auto &as_em =
        static_cast<edge_membership<Index> &>(static_cast<b_base_t &>(*this));
    as_em.build(edges, n_unique_ids, eo);
    switch (eo) {
    case tf::edge_orientation::forward:
      return tf::parallel_copy(
          tf::make_indirect_range(
              as_em.data_buffer(),
              tf::make_mapped_range(edges, [](const auto &r) { return r[1]; })),
          as_em.data_buffer());
    case tf::edge_orientation::reverse:
      return tf::parallel_copy(
          tf::make_indirect_range(
              as_em.data_buffer(),
              tf::make_mapped_range(edges, [](const auto &r) { return r[0]; })),
          as_em.data_buffer());
    case tf::edge_orientation::bidirectional:
      tf::parallel_for_each(
          tf::enumerate(as_em),
          [&](auto pair) {
            auto &&[id, block] = pair;
            for (auto &edge_id : block) {
              const auto &edge = edges[edge_id];
              edge_id = edge[edge[0] == Index(id)];
            }
          },
          tf::checked);
    }
  }

  /// @brief Build from segments with specified orientation.
  /// @tparam Policy The segments policy type.
  /// @param segments The segments range.
  /// @param eo The edge orientation mode (default: bidirectional).
  template <typename Policy>
  auto build(const tf::segments<Policy> &segments,
             tf::edge_orientation eo = tf::edge_orientation::bidirectional) {
    build(segments.edges(), segments.points().size(), eo);
  }
};

} // namespace tf
