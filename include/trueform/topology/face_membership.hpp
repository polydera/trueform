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
#include "../core/algorithm/reduce.hpp"
#include "../core/faces.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/polygons.hpp"
#include "../core/views/mapped_range.hpp"
#include "./face_membership_like.hpp"
#include "./structures/compute_face_membership.hpp"

namespace tf {

/// @ingroup topology_connectivity
/// @brief Maps each vertex to the faces that contain it.
///
/// This is the fundamental connectivity structure for mesh topology operations.
/// For each vertex index, it stores the list of face indices that reference
/// that vertex. This enables efficient traversal of a vertex's incident faces.
///
/// Build methods:
/// - From a @ref tf::polygons range (deduces sizes automatically)
/// - From @ref tf::faces with explicit counts
///
/// @tparam Index The integer type for vertex and face indices.
template <typename Index>
class face_membership
    : public face_membership_like<offset_block_buffer<Index, Index>> {
  using base_t = face_membership_like<offset_block_buffer<Index, Index>>;

public:
  face_membership() = default;

  /// @brief Construct and build from a polygons range.
  /// @tparam Policy The polygons policy type.
  /// @param polygons The polygons range to build from.
  template <typename Policy> face_membership(const polygons<Policy> &polygons) {
    build(polygons);
  }

  /// @brief Build from faces with explicit size parameters.
  /// @tparam Policy The faces policy type.
  /// @param faces The faces range.
  /// @param n_unique_ids The number of unique vertex ids.
  /// @param total_size The total number of vertex references across all faces.
  template <typename Policy>
  auto build(const tf::faces<Policy> &faces, std::size_t n_unique_ids,
             std::size_t total_size) -> void {
    base_t::offsets_buffer().allocate(n_unique_ids + 1);
    base_t::data_buffer().allocate(total_size);
    topology::compute_face_membership(faces, base_t::offsets_buffer(),
                                      base_t::data_buffer());
  }

  /// @brief Build from faces with automatic total size computation.
  ///
  /// Computes the total size from the faces range. For fixed-size polygons,
  /// this is n_gons * faces.size(). For variable-size, it sums face sizes.
  ///
  /// @tparam Policy The faces policy type.
  /// @param faces The faces range.
  /// @param n_unique_ids The number of unique vertex ids.
  template <typename Policy>
  auto build(const tf::faces<Policy> &faces, std::size_t n_unique_ids) -> void {
    constexpr auto n_gons = tf::static_size_v<decltype(faces[0])>;
    if constexpr (n_gons != tf::dynamic_size) {
      build(faces, n_unique_ids, n_gons * faces.size());
    } else {
      auto sizes = tf::make_mapped_range(
          faces, [](const auto &face) { return face.size(); });
      auto total_size = tf::reduce(
          sizes, [](auto a, auto b) { return a + b; }, std::size_t{0},
          tf::checked);
      build(faces, n_unique_ids, total_size);
    }
  }

  /// @brief Build from a polygons range.
  ///
  /// Automatically deduces the number of vertices and total size from the
  /// polygons. Supports both fixed-size and variable-size polygons.
  ///
  /// @tparam Policy The polygons policy type.
  /// @param polygons The polygons range to build from.
  template <typename Policy>
  auto build(const polygons<Policy> &polygons) -> void {
    return build(polygons.faces(), polygons.points().size());
  }
};

} // namespace tf
