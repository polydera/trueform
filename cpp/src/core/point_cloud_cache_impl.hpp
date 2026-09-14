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

#include "trueform/cpp/core/point_cloud_cache.hpp"

#include "./carrier_arrays.hpp"

#include "trueform/cpp/core/detail/fill_guard.hpp"
#include "trueform/spatial/tree_config.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace tf::cpp {

template <typename Real, std::size_t Dims>
point_cloud_cache<Real, Dims>::point_cloud_cache(
    const point_cloud_cache &other) {
  copy_from(other);
}

template <typename Real, std::size_t Dims>
point_cloud_cache<Real, Dims>::point_cloud_cache(
    point_cloud_cache &&other) noexcept {
  move_from(std::move(other));
}

template <typename Real, std::size_t Dims>
auto point_cloud_cache<Real, Dims>::operator=(const point_cloud_cache &other)
    -> point_cloud_cache & {
  if (this != &other)
    copy_from(other);
  return *this;
}

template <typename Real, std::size_t Dims>
auto point_cloud_cache<Real, Dims>::operator=(
    point_cloud_cache &&other) noexcept -> point_cloud_cache & {
  if (this != &other)
    move_from(std::move(other));
  return *this;
}

/// A copy shares nothing, so what it carries is what it can own outright. The
/// tree holds views into its own storage, so a copy builds one when it is asked
/// for one.
template <typename Real, std::size_t Dims>
auto point_cloud_cache<Real, Dims>::copy_from(const point_cloud_cache &other)
    -> void {
  _vertex_link = other._vertex_link.deep_copy();
  _tree.reset();
  _points_generation = other._points_generation;
  _tree_points_stamp = 0;
  _vertex_link_points_stamp = other._vertex_link_points_stamp;
  _tree_build_count = 0;
}

template <typename Real, std::size_t Dims>
auto point_cloud_cache<Real, Dims>::move_from(point_cloud_cache &&other)
    -> void {
  _vertex_link = std::move(other._vertex_link);
  _tree = std::move(other._tree);
  _points_generation = other._points_generation;
  _tree_points_stamp = other._tree_points_stamp;
  _vertex_link_points_stamp = other._vertex_link_points_stamp;
  _tree_build_count = other._tree_build_count;
}

template <typename Real, std::size_t Dims>
auto point_cloud_cache<Real, Dims>::points_changed() -> void {
  ++_points_generation;
}

template <typename Real, std::size_t Dims>
auto point_cloud_cache<Real, Dims>::points_generation() const -> std::uint64_t {
  return _points_generation;
}

template <typename Real, std::size_t Dims>
auto point_cloud_cache<Real, Dims>::assert_is_current(
    const geometry_type &geometry) const -> void {
  static_cast<void>(geometry);
  assert(geometry.points_stamp == _points_generation &&
         "trueform cache: a point cloud assembled before a stated change is "
         "reading through arrays this cache no longer answers for. Assemble "
         "it again after points_changed()");
}

template <typename Real, std::size_t Dims>
auto point_cloud_cache<Real, Dims>::ensure_tree(const geometry_type &geometry)
    -> void {
  if (is_tree_fresh(geometry))
    return;
  detail::fill_guard guard(_filling);
  _tree = std::make_shared<tf::aabb_tree<std::int32_t, Real, Dims>>(
      geometry.points(), tf::config_tree(4, 12));
  _tree_points_stamp = geometry.points_stamp;
  ++_tree_build_count;
}

template <typename Real, std::size_t Dims>
auto point_cloud_cache<Real, Dims>::tree_handle(const geometry_type &geometry)
    -> std::shared_ptr<tf::aabb_tree<std::int32_t, Real, Dims>> {
  assert_is_current(geometry);
  ensure_tree(geometry);
  return _tree;
}

template <typename Real, std::size_t Dims>
auto point_cloud_cache<Real, Dims>::vertex_link_handle(
    const geometry_type &geometry) const
    -> offset_blocked_buffer<std::int32_t, std::int32_t> {
  assert_is_current(geometry);
  if (_vertex_link_points_stamp != geometry.points_stamp)
    return {};
  return _vertex_link;
}

template <typename Real, std::size_t Dims>
auto point_cloud_cache<Real, Dims>::is_vertex_link_fresh(
    const geometry_type &geometry) const -> bool {
  return _vertex_link.is_valid() &&
         _vertex_link_points_stamp == geometry.points_stamp;
}

template <typename Real, std::size_t Dims>
auto point_cloud_cache<Real, Dims>::set_vertex_link(
    offset_blocked_buffer<std::int32_t, std::int32_t> vertex_link,
    const geometry_type &geometry) -> void {
  assert_is_current(geometry);
  carrier::require_offset_blocks(
      vertex_link, static_cast<int>(geometry.number_of_points()),
      static_cast<int>(geometry.number_of_points()), "vertex link");
  detail::fill_guard guard(_filling);
  _vertex_link = std::move(vertex_link);
  _vertex_link_points_stamp = geometry.points_stamp;
}

template <typename Real, std::size_t Dims>
auto point_cloud_cache<Real, Dims>::is_tree_built() const -> bool {
  return static_cast<bool>(_tree);
}

template <typename Real, std::size_t Dims>
auto point_cloud_cache<Real, Dims>::is_tree_fresh(
    const geometry_type &geometry) const -> bool {
  return _tree && _tree_points_stamp == geometry.points_stamp;
}

template <typename Real, std::size_t Dims>
auto point_cloud_cache<Real, Dims>::tree_build_count() const -> std::uint64_t {
  return _tree_build_count;
}

} // namespace tf::cpp
