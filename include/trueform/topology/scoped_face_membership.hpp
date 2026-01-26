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
#include "../core/offset_block_buffer.hpp"
#include "../core/polygons.hpp"
#include "./scoped_id.hpp"
#include "./structures/compute_face_membership.hpp"

namespace tf {

/// @ingroup topology_connectivity
/// @brief Face membership with scoped IDs for position-in-face tracking.
///
/// Like @ref tf::face_membership, but stores @ref tf::scoped_id pairs
/// that include both the face index and the vertex's position within
/// that face. This is useful when you need to know not just which faces
/// contain a vertex, but also where in each face the vertex appears.
///
/// @tparam Index The integer type for face indices.
/// @tparam SubIndex The integer type for position within face.
template <typename Index, typename SubIndex = Index>
class scoped_face_membership
    : public offset_block_buffer<Index, tf::scoped_id<Index, SubIndex>> {
  using base_t = offset_block_buffer<Index, tf::scoped_id<Index, SubIndex>>;

public:
  /// @brief Build from blocks with explicit size parameters.
  /// @tparam Range The blocks range type.
  /// @param blocks The blocks range.
  /// @param n_unique_ids The number of unique vertex ids.
  /// @param total_size The total number of vertex references.
  template <typename Range>
  auto build(const Range &blocks, std::size_t n_unique_ids,
             std::size_t total_size) -> void {
    base_t::offsets_buffer().allocate(n_unique_ids + 1);
    base_t::data_buffer().allocate(total_size);
    topology::compute_face_membership(
        blocks, base_t::offsets_buffer(), base_t::data_buffer(),
        [](auto sub_id, auto block_id) {
          return tf::scoped_id{Index(block_id), SubIndex(sub_id)};
        });
  }

  /// @brief Build from a fixed-size polygons range.
  ///
  /// Only works with fixed-size polygons (compile-time known size).
  ///
  /// @tparam Policy The polygons policy type.
  /// @param polygons The polygons range to build from.
  template <typename Policy>
  auto build(const polygons<Policy> &polygons) -> void {
    auto n_unique_ids = polygons.points().size();
    constexpr auto n_gons = tf::static_size_v<decltype(polygons[0])>;
    static_assert(n_gons != tf::dynamic_size);
    build(polygons, n_unique_ids, n_gons * polygons.size());
  }
};

} // namespace tf
