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
#include "trueform/core/transformation_view.hpp"
#include "trueform/cpp/core/detail/reading_views.hpp"
#include "trueform/cpp/core/identity_transformation.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/cpp/core/point_cloud_cache.hpp"
#include "trueform/cpp/core/point_cloud_geometry.hpp"
#include "trueform/spatial/aabb_tree.hpp"
#include "trueform/spatial/policy/tree.hpp"
#include "trueform/topology/vertex_link_like.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace tf::cpp {

/// @brief A point cloud: points a caller holds, a cache that remembers them,
/// and this instance's frame.
///
/// The mesh's contract, for the point carrier — see
/// `trueform/cpp/core/mesh.hpp`. The normals are part of the geometry here: a
/// cloud has no faces, so they are a per-point attribute of the very points a
/// reading stands on, and they are stated with them.
template <typename Real, std::size_t Dims = 3> class point_cloud {
  static_assert(Dims == 2 || Dims == 3, "point_cloud Dims must be 2 or 3");
  static_assert(matrix_carries_points_v<Real, Dims>,
                "this (real, dims) is not in the built C++ facade matrix; see "
                "TF_CPP_REALS / TF_CPP_DIMS");

public:
  using real_type = Real;
  using geometry_type = point_cloud_geometry<Real, Dims>;
  using points_type = typename geometry_type::points_type;
  using normals_type = typename geometry_type::normals_type;
  using frame_type = tf::transformation_view<const Real, Dims>;
  using cache_type = point_cloud_cache<Real, Dims>;

  using tree_type = tf::aabb_tree<std::int32_t, Real, Dims>;

  point_cloud() = delete;
  /// @brief The assembly.
  template <
      typename Points,
      std::enable_if_t<detail::reads_blocks_v<Points, Real, Dims>, int> = 0>
  point_cloud(const Points &points, cache_type &cache,
              frame_type frame = identity_transformation_view<Real, Dims>(),
              std::shared_ptr<const void> keepalive = {})
      : _geometry(reading_of(points, cache.points_generation())), _frame(frame),
        _cache(&cache), _keepalive(std::move(keepalive)) {}
  /// @overload With the normals these points carry.
  template <typename Points, typename Normals,
            std::enable_if_t<detail::reads_blocks_v<Points, Real, Dims> &&
                                 detail::reads_blocks_v<Normals, Real, Dims>,
                             int> = 0>
  point_cloud(const Points &points, const Normals &normals, cache_type &cache,
              frame_type frame = identity_transformation_view<Real, Dims>(),
              std::shared_ptr<const void> keepalive = {})
      : _geometry(reading_of(points, normals, cache.points_generation())),
        _frame(frame), _cache(&cache), _keepalive(std::move(keepalive)) {}

  auto points() const -> points_type { return _geometry.points(); }
  auto frame() const -> frame_type { return _frame; }
  auto geometry() const -> const geometry_type & { return _geometry; }
  auto number_of_points() const -> std::size_t {
    return _geometry.number_of_points();
  }

  /// A cloud has no faces, so it has one set of normals: its points'. They ride
  /// the reading, so they name the points this cloud holds and not whatever the
  /// caller holds by now.
  auto has_normals() const -> bool { return _geometry.has_normals(); }
  auto normals() const { return _geometry.normals(); }

  auto tree() const -> const tree_type & {
    if (!_tree)
      _tree = _cache->tree_handle(_geometry);
    return *_tree;
  }
  auto vertex_link() const {
    if (!_vertex_link.is_valid())
      _vertex_link = _cache->vertex_link_handle(_geometry);
    return tf::make_vertex_link_like(std::as_const(_vertex_link).make_range());
  }

  /// @brief The same cloud, read in its own frame.
  auto at_identity() const -> point_cloud {
    return point_cloud(_geometry, identity_transformation_view<Real, Dims>(),
                       *_cache, _keepalive);
  }

  /// The points, the tree and the frame: what a spatial query takes. It names
  /// ONE structure, the tree.
  auto form() const {
    return _geometry.points() | tf::tag(tree()) | tf::tag(_frame);
  }

  auto cache() const -> cache_type & { return *_cache; }

private:
  point_cloud(geometry_type geometry, frame_type frame, cache_type &cache,
              std::shared_ptr<const void> keepalive)
      : _geometry(std::move(geometry)), _frame(frame), _cache(&cache),
        _keepalive(std::move(keepalive)) {}

  /// The flat arrays a points and a normals view stand on: a reading is flat.
  template <typename Points>
  static auto reading_of(const Points &points, std::uint64_t points_stamp)
      -> geometry_type {
    return {detail::const_range_of(points.begin().base_iter(),
                                   points.end().base_iter()),
            {},
            points_stamp};
  }

  template <typename Points, typename Normals>
  static auto reading_of(const Points &points, const Normals &normals,
                         std::uint64_t points_stamp) -> geometry_type {
    return {detail::const_range_of(points.begin().base_iter(),
                                   points.end().base_iter()),
            detail::const_range_of(normals.begin().base_iter(),
                                   normals.end().base_iter()),
            points_stamp};
  }

  geometry_type _geometry;
  frame_type _frame;
  /// A structure is asked for, not taken: a body that never queries one pays
  /// nothing for it. Retaining what it hands back is not optional — a tag keeps
  /// views into its arrays, and the cache REPLACES the whole of one when it
  /// rebuilds it, so a reference alone would dangle the moment another reading
  /// asks. One slot each, filled once, for this cloud's lifetime.
  mutable std::shared_ptr<const tree_type> _tree;
  mutable offset_blocked_buffer<std::int32_t, std::int32_t> _vertex_link;
  cache_type *_cache;
  std::shared_ptr<const void> _keepalive;
};

} // namespace tf::cpp
