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
#include "../core/blocked_buffer.hpp"
#include "../core/faces.hpp"
#include "../core/index_map.hpp"
#include "../core/none.hpp"
#include "../core/point.hpp"
#include "../core/points.hpp"
#include "../exact/resolve_int_type.hpp"
#include "../reindex/return_index_map.hpp"
#include "./cdt/build_unconstrained_delaunay.hpp"
#include "./cdt/clear_unconstrained_delaunay.hpp"
#include "./cdt/delaunay_execution_policy.hpp"
#include "./cdt/delaunay_vertex_policy.hpp"
#include "./cdt/unconstrained_delaunay_owner.hpp"
#include <cstddef>
#include <utility>

namespace tf {

/// @ingroup topology
/// @brief Triangle-only, unconstrained Delaunay triangulation.
///
/// Unlike constrained_delaunay_triangulator, this operation deliberately
/// retains no editable constraint state, adjacency, or region labels. Exact
/// duplicate coordinates retain the lowest input index.
///
/// `VertexPolicy` selects the identity written to each face corner. The default
/// `original_input_vertex_policy` makes faces index the caller's point array.
/// `compact_topology_vertex_policy` instead makes faces index the exact-welded
/// compact vertices exposed by `unique_input_id()` and
/// `converted_unique_point()`.
///
/// @tparam Index Integer type used for point and topology IDs.
/// @tparam Coord Input coordinate type and the type returned by
///   `converted_unique_point()`.
/// @tparam Int Integer type used by exact predicates; defaults per Coord
///   via @ref tf::exact::resolve_int_type (int32 for float, int64 for
///   double, identity for integral Coord).
/// @tparam VertexPolicy Selects original-input or compact face vertex IDs.
/// @tparam ExecutionPolicy Selects serial or separated-domain parallel
///   scheduling for preparation, topology construction, and face emission.
template <typename Index = int, typename Coord = float,
          typename Int = tf::exact::resolve_int_type<tf::none_t, Coord>,
          typename VertexPolicy =
              tf::topology::cdt::original_input_vertex_policy,
          typename ExecutionPolicy =
              tf::topology::cdt::serial_delaunay_execution_policy>
class unconstrained_delaunay_triangulator {
  using owner_type =
      tf::topology::cdt::unconstrained_delaunay_owner<Index, Coord, Int,
                                                      VertexPolicy>;

  auto owner() -> owner_type & { return _owner; }
  auto owner() const -> const owner_type & { return _owner; }

  owner_type _owner;

public:
  /// Clear all output and build state while retaining buffer capacity.
  auto clear() -> void {
    tf::topology::cdt::clear_unconstrained_delaunay(owner());
  }

  /// Build faces in the vertex identity selected by `VertexPolicy`.
  /// Returns false when fewer than three input points are supplied or the
  /// selected `Index` cannot address the topology. Three or more inputs that
  /// weld to fewer than three unique positions are accepted and produce no
  /// faces.
  template <typename PointsPolicy>
  auto build(const tf::points<PointsPolicy> &points) -> bool {
    return tf::topology::cdt::build_unconstrained_delaunay<false,
                                                           ExecutionPolicy>(
        owner(), points);
  }

  /// Build and retain an input-to-compact-topology index map.
  ///
  /// The map always names compact topology vertices, independently of
  /// `VertexPolicy`. With `compact_topology_vertex_policy`, `f()[input]`
  /// therefore names the same vertex identity used by `faces()`, and
  /// `kept_ids()[vertex]` names its lowest-ID input representative. With the
  /// default original-input policy, faces continue to use original input IDs.
  template <typename PointsPolicy>
  auto build(const tf::points<PointsPolicy> &points, tf::return_index_map_t)
      -> bool {
    return tf::topology::cdt::build_unconstrained_delaunay<true,
                                                           ExecutionPolicy>(
        owner(), points);
  }

  /// Move out the compact index map produced by the tagged `build()` overload.
  /// The returned map remains complete when three or more inputs weld to fewer
  /// than three unique points. This accessor empties the retained map.
  auto take_index_map() -> tf::index_map_buffer<Index> {
    return std::move(owner()._index_map);
  }

  /// Faces in the vertex identity selected by `VertexPolicy`.
  auto faces() const { return tf::make_faces(owner()._faces); }

  /// Owning face carrier for code that needs direct buffer access.
  auto faces_buffer() const -> const tf::blocked_buffer<Index, 3> & {
    return owner()._faces;
  }

  /// Move out the face carrier. This empties `faces()` until the next build.
  auto take_faces() -> tf::blocked_buffer<Index, 3> {
    return std::move(owner()._faces);
  }

  /// Number of exact-welded compact topology vertices.
  auto n_unique_points() const -> std::size_t { return owner()._sites.size(); }

  /// Lowest original input ID retained for a compact topology vertex.
  auto unique_input_id(std::size_t vertex) const -> Index {
    return owner()._sites[vertex].output;
  }

  /// A compact topology vertex deconverted to `Coord`.
  auto converted_unique_point(std::size_t vertex) const {
    const auto &site = owner()._sites[vertex];
    return owner()._converter.deconvert(tf::make_point(site.x, site.y));
  }
};

} // namespace tf
