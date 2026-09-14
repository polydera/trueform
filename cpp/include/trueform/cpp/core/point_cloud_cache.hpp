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

#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/cpp/core/point_cloud_geometry.hpp"
#include "trueform/spatial/aabb_tree.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace tf::cpp {

/// @brief What is known about a point cloud's shape, remembered.
///
/// The mesh cache's contract, for the point carrier — see
/// `trueform/cpp/core/cache.hpp`. The vertex link is where these
/// points are near each other and has no builder to say it again, so a reading
/// whose points have moved simply does not have one.
template <typename Real, std::size_t Dims = 3> class point_cloud_cache {
  static_assert(Dims == 2 || Dims == 3,
                "point_cloud_cache Dims must be 2 or 3");
  static_assert(matrix_carries_points_v<Real, Dims>,
                "this (real, dims) is not in the built C++ facade matrix; see "
                "TF_CPP_REALS / TF_CPP_DIMS");

  using geometry_type = point_cloud_geometry<Real, Dims>;

  offset_blocked_buffer<std::int32_t, std::int32_t> _vertex_link;
  std::shared_ptr<tf::aabb_tree<std::int32_t, Real, Dims>> _tree;

  std::uint64_t _points_generation = 1;

  std::uint64_t _tree_points_stamp = 0;
  std::uint64_t _vertex_link_points_stamp = 0;
  std::uint64_t _tree_build_count = 0;

  mutable std::atomic<bool> _filling{false};

public:
  point_cloud_cache() = default;
  point_cloud_cache(const point_cloud_cache &other);
  point_cloud_cache(point_cloud_cache &&other) noexcept;
  auto operator=(const point_cloud_cache &other) -> point_cloud_cache &;
  auto operator=(point_cloud_cache &&other) noexcept -> point_cloud_cache &;

  /// @brief The caller states the change; the cache answers for it.
  auto points_changed() -> void;
  auto points_generation() const -> std::uint64_t;

  auto tree_handle(const geometry_type &geometry)
      -> std::shared_ptr<tf::aabb_tree<std::int32_t, Real, Dims>>;

  /// @brief The link, retained: a rebuild replaces the whole of it, so a
  /// reader that keeps views into its arrays keeps the link itself.
  auto vertex_link_handle(const geometry_type &geometry) const
      -> offset_blocked_buffer<std::int32_t, std::int32_t>;
  auto is_vertex_link_fresh(const geometry_type &geometry) const -> bool;
  auto
  set_vertex_link(offset_blocked_buffer<std::int32_t, std::int32_t> vertex_link,
                  const geometry_type &geometry) -> void;

  auto is_tree_built() const -> bool;
  auto is_tree_fresh(const geometry_type &geometry) const -> bool;
  auto tree_build_count() const -> std::uint64_t;

private:
  auto copy_from(const point_cloud_cache &other) -> void;
  auto move_from(point_cloud_cache &&other) -> void;
  auto assert_is_current(const geometry_type &geometry) const -> void;
  auto ensure_tree(const geometry_type &geometry) -> void;
};

#define TF_CPP_EXTERN_POINT_CLOUD_STRUCTURE(Real, Dims)                        \
  extern template class point_cloud_cache<Real, Dims>

TF_CPP_MATRIX_FOR_EACH_REAL_DIMS(TF_CPP_EXTERN_POINT_CLOUD_STRUCTURE)

#undef TF_CPP_EXTERN_POINT_CLOUD_STRUCTURE

} // namespace tf::cpp
