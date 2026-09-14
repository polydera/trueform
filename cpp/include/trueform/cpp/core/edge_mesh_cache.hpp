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

#include "trueform/cpp/core/edge_mesh_geometry.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/spatial/aabb_tree.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace tf::cpp {

/// @brief What is known about an edge mesh's shape, remembered.
///
/// The mesh cache's contract, for the edge carrier — see
/// `trueform/cpp/core/cache.hpp`.
template <typename Index, typename Real, std::size_t Dims = 3>
class edge_mesh_cache {
  static_assert(is_supported_index_v<Index> &&
                    std::is_same_v<Index, std::remove_cv_t<Index>>,
                "edge_mesh_cache Index must be an unqualified supported "
                "index type");
  static_assert(Dims == 2 || Dims == 3,
                "edge_mesh_cache Dims must be 2 or 3");
  static_assert(matrix_carries_v<Index, Real, Dims>,
                "this (index, real, dims) is not in the built C++ facade "
                "matrix; see TF_CPP_INDICES / TF_CPP_REALS / TF_CPP_DIMS");

  using geometry_type = edge_mesh_geometry<Index, Real, Dims>;

  offset_blocked_buffer<Index, Index> _edge_membership;
  offset_blocked_buffer<Index, Index> _vertex_link;
  std::shared_ptr<tf::aabb_tree<Index, Real, Dims>> _tree;

  std::uint64_t _edges_generation = 1;
  std::uint64_t _points_generation = 1;

  std::uint64_t _indices_edges_stamp = 0;
  int _indices_point_count = -1;
  std::uint64_t _tree_edges_stamp = 0;
  std::uint64_t _tree_points_stamp = 0;
  std::uint64_t _edge_membership_edges_stamp = 0;
  int _edge_membership_point_count = -1;
  std::uint64_t _vertex_link_edges_stamp = 0;
  int _vertex_link_point_count = -1;

  std::uint64_t _tree_build_count = 0;
  std::uint64_t _edge_membership_build_count = 0;
  std::uint64_t _vertex_link_build_count = 0;

  mutable std::atomic<bool> _filling{false};

public:
  edge_mesh_cache() = default;
  edge_mesh_cache(const edge_mesh_cache &other);
  edge_mesh_cache(edge_mesh_cache &&other) noexcept;
  auto operator=(const edge_mesh_cache &other) -> edge_mesh_cache &;
  auto operator=(edge_mesh_cache &&other) noexcept -> edge_mesh_cache &;

  /// @brief The caller states the change; the cache answers for it.
  auto edges_changed() -> void;
  auto points_changed() -> void;
  auto edges_generation() const -> std::uint64_t;
  auto points_generation() const -> std::uint64_t;

  /// @brief Every structure is handed back RETAINED.
  ///
  /// A rebuild replaces the whole structure, so a reader that keeps views into
  /// its arrays keeps the structure itself: a handle shares the storage this
  /// one stands on and outlives whatever another reading asks for next.
  auto tree_handle(const geometry_type &geometry)
      -> std::shared_ptr<tf::aabb_tree<Index, Real, Dims>>;
  auto edge_membership_handle(const geometry_type &geometry)
      -> offset_blocked_buffer<Index, Index>;
  auto vertex_link_handle(const geometry_type &geometry)
      -> offset_blocked_buffer<Index, Index>;

  /// @brief Refuse an edge index the points do not have.
  auto require_indices(const geometry_type &geometry) -> void;

  /// @brief State a structure a caller already holds.
  auto set_edge_membership(offset_blocked_buffer<Index, Index> edge_membership,
                           const geometry_type &geometry) -> void;
  auto set_vertex_link(offset_blocked_buffer<Index, Index> vertex_link,
                       const geometry_type &geometry) -> void;

  auto is_tree_built() const -> bool;
  auto is_tree_fresh(const geometry_type &geometry) const -> bool;
  auto is_edge_membership_built() const -> bool;
  auto is_edge_membership_fresh(const geometry_type &geometry) const -> bool;
  auto is_vertex_link_built() const -> bool;
  auto is_vertex_link_fresh(const geometry_type &geometry) const -> bool;

  auto tree_build_count() const -> std::uint64_t;
  auto edge_membership_build_count() const -> std::uint64_t;
  auto vertex_link_build_count() const -> std::uint64_t;

private:
  auto copy_from(const edge_mesh_cache &other) -> void;
  auto move_from(edge_mesh_cache &&other) -> void;
  auto assert_is_current(const geometry_type &geometry) const -> void;
  auto ensure_indices(const geometry_type &geometry) -> void;
  auto ensure_tree(const geometry_type &geometry) -> void;
  auto ensure_edge_membership(const geometry_type &geometry) -> void;
  auto ensure_vertex_link(const geometry_type &geometry) -> void;
};

#define TF_CPP_EXTERN_EDGE_MESH_STRUCTURE(Index, Real, Dims)                   \
  extern template class edge_mesh_cache<Index, Real, Dims>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS(TF_CPP_EXTERN_EDGE_MESH_STRUCTURE)

#undef TF_CPP_EXTERN_EDGE_MESH_STRUCTURE

} // namespace tf::cpp
