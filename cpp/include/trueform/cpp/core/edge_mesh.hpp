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

#include "trueform/core/policy/frame.hpp"
#include "trueform/core/segments.hpp"
#include "trueform/core/transformation_view.hpp"
#include "trueform/cpp/core/detail/reading_views.hpp"
#include "trueform/cpp/core/edge_mesh_cache.hpp"
#include "trueform/cpp/core/edge_mesh_geometry.hpp"
#include "trueform/cpp/core/identity_transformation.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/spatial/aabb_tree.hpp"
#include "trueform/spatial/policy/tree.hpp"
#include "trueform/topology/edge_membership_like.hpp"
#include "trueform/topology/policy/edge_membership.hpp"
#include "trueform/topology/vertex_link_like.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace tf::cpp {

/// @brief An edge mesh: segments a caller holds, a cache that remembers them,
/// and this instance's frame.
///
/// The mesh's contract, for the edge carrier — see
/// `trueform/cpp/core/mesh.hpp`.
template <typename Index, typename Real, std::size_t Dims = 3> class edge_mesh {
  static_assert(is_supported_index_v<Index> &&
                    std::is_same_v<Index, std::remove_cv_t<Index>>,
                "edge_mesh Index must be an unqualified supported index type");
  static_assert(Dims == 2 || Dims == 3, "edge_mesh Dims must be 2 or 3");
  static_assert(matrix_carries_v<Index, Real, Dims>,
                "this (index, real, dims) is not in the built C++ facade "
                "matrix; see TF_CPP_INDICES / TF_CPP_REALS / TF_CPP_DIMS");

public:
  using index_type = Index;
  using real_type = Real;
  using geometry_type = edge_mesh_geometry<Index, Real, Dims>;
  using edges_type = typename geometry_type::edges_type;
  using points_type = typename geometry_type::points_type;
  using frame_type = tf::transformation_view<const Real, Dims>;
  using cache_type = edge_mesh_cache<Index, Real, Dims>;

  using tree_type = tf::aabb_tree<Index, Real, Dims>;

  edge_mesh() = delete;
  /// @brief The assembly.
  template <typename Edges, typename Points,
            std::enable_if_t<detail::reads_blocks_v<Edges, Index, 2> &&
                                 detail::reads_blocks_v<Points, Real, Dims>,
                             int> = 0>
  edge_mesh(const Edges &edges, const Points &points, cache_type &cache,
            frame_type frame = identity_transformation_view<Real, Dims>(),
            std::shared_ptr<const void> keepalive = {})
      : _geometry(reading_of(edges, points, cache.edges_generation(),
                             cache.points_generation())),
        _frame(frame), _cache(&cache), _keepalive(std::move(keepalive)) {}

  auto edges() const -> edges_type { return _geometry.edges(); }
  auto points() const -> points_type { return _geometry.points(); }
  auto segments() const {
    return tf::make_segments(_geometry.edges(), _geometry.points());
  }
  auto frame() const -> frame_type { return _frame; }
  auto geometry() const -> const geometry_type & { return _geometry; }

  auto number_of_edges() const -> std::size_t {
    return _geometry.number_of_edges();
  }
  auto number_of_points() const -> std::size_t {
    return _geometry.number_of_points();
  }

  auto tree() const -> const tree_type & {
    if (!_tree)
      _tree = _cache->tree_handle(_geometry);
    return *_tree;
  }
  auto edge_membership() const {
    if (!_edge_membership.is_valid())
      _edge_membership = _cache->edge_membership_handle(_geometry);
    return tf::make_edge_membership_like(
        std::as_const(_edge_membership).make_range());
  }
  auto vertex_link() const {
    if (!_vertex_link.is_valid())
      _vertex_link = _cache->vertex_link_handle(_geometry);
    return tf::make_vertex_link_like(std::as_const(_vertex_link).make_range());
  }
  /// @brief Refuse an edge index these points do not have.
  ///
  /// The cache owns the fact and remembers it for this reading, so a second
  /// ask costs nothing and no body carries its own scan.
  auto require_indices() const -> void { _cache->require_indices(_geometry); }

  /// @brief The same edge mesh, read in its own frame.
  auto at_identity() const -> edge_mesh {
    return edge_mesh(_geometry, identity_transformation_view<Real, Dims>(),
                     *_cache, _keepalive);
  }

  /// The segments, the tree and the frame: what a spatial query takes. It
  /// names ONE structure, the tree.
  auto form() const { return segments() | tf::tag(tree()) | tf::tag(_frame); }

  /// The same, with the topology a walk needs: the tree and the edge
  /// membership.
  auto topology_form() const {
    return segments() | tf::tag(tree()) |
           tf::tag_edge_membership(edge_membership()) | tf::tag(_frame);
  }

  auto cache() const -> cache_type & { return *_cache; }

private:
  edge_mesh(geometry_type geometry, frame_type frame, cache_type &cache,
            std::shared_ptr<const void> keepalive)
      : _geometry(std::move(geometry)), _frame(frame), _cache(&cache),
        _keepalive(std::move(keepalive)) {}

  /// The flat arrays an edges and a points view stand on: a reading is flat.
  template <typename Edges, typename Points>
  static auto reading_of(const Edges &edges, const Points &points,
                         std::uint64_t edges_stamp, std::uint64_t points_stamp)
      -> geometry_type {
    return {detail::const_range_of(edges.begin().base_iter(),
                                   edges.end().base_iter()),
            detail::const_range_of(points.begin().base_iter(),
                                   points.end().base_iter()),
            edges_stamp, points_stamp};
  }

  geometry_type _geometry;
  frame_type _frame;
  /// A structure is asked for, not taken, and RETAINED the first time it is: a
  /// tag keeps views into its arrays and the cache replaces the whole of one
  /// when it rebuilds it.
  mutable std::shared_ptr<const tree_type> _tree;
  mutable offset_blocked_buffer<Index, Index> _edge_membership;
  mutable offset_blocked_buffer<Index, Index> _vertex_link;
  cache_type *_cache;
  std::shared_ptr<const void> _keepalive;
};

} // namespace tf::cpp
