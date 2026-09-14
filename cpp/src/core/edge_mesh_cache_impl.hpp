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

#include "trueform/cpp/core/edge_mesh_cache.hpp"

#include "./carrier_arrays.hpp"
#include "./require_point_indices.hpp"

#include "trueform/core/segments.hpp"
#include "trueform/cpp/core/detail/fill_guard.hpp"
#include "trueform/spatial/tree_config.hpp"
#include "trueform/topology/edge_membership.hpp"
#include "trueform/topology/vertex_link.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tf::cpp {

template <typename Index, typename Real, std::size_t Dims>
edge_mesh_cache<Index, Real, Dims>::edge_mesh_cache(
    const edge_mesh_cache &other) {
  copy_from(other);
}

template <typename Index, typename Real, std::size_t Dims>
edge_mesh_cache<Index, Real, Dims>::edge_mesh_cache(
    edge_mesh_cache &&other) noexcept {
  move_from(std::move(other));
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::operator=(const edge_mesh_cache &other)
    -> edge_mesh_cache & {
  if (this != &other)
    copy_from(other);
  return *this;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::operator=(
    edge_mesh_cache &&other) noexcept -> edge_mesh_cache & {
  if (this != &other)
    move_from(std::move(other));
  return *this;
}

/// A copy shares nothing, so what it carries is what it can own outright. The
/// tree holds views into its own storage, so a copy builds one when it is asked
/// for one.
template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::copy_from(const edge_mesh_cache &other)
    -> void {
  _edge_membership = other._edge_membership.deep_copy();
  _vertex_link = other._vertex_link.deep_copy();
  _tree.reset();
  _edges_generation = other._edges_generation;
  _points_generation = other._points_generation;
  _indices_edges_stamp = other._indices_edges_stamp;
  _indices_point_count = other._indices_point_count;
  _tree_edges_stamp = 0;
  _tree_points_stamp = 0;
  _edge_membership_edges_stamp = other._edge_membership_edges_stamp;
  _edge_membership_point_count = other._edge_membership_point_count;
  _vertex_link_edges_stamp = other._vertex_link_edges_stamp;
  _vertex_link_point_count = other._vertex_link_point_count;
  _tree_build_count = 0;
  _edge_membership_build_count = 0;
  _vertex_link_build_count = 0;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::move_from(edge_mesh_cache &&other)
    -> void {
  _edge_membership = std::move(other._edge_membership);
  _vertex_link = std::move(other._vertex_link);
  _tree = std::move(other._tree);
  _edges_generation = other._edges_generation;
  _points_generation = other._points_generation;
  _indices_edges_stamp = other._indices_edges_stamp;
  _indices_point_count = other._indices_point_count;
  _tree_edges_stamp = other._tree_edges_stamp;
  _tree_points_stamp = other._tree_points_stamp;
  _edge_membership_edges_stamp = other._edge_membership_edges_stamp;
  _edge_membership_point_count = other._edge_membership_point_count;
  _vertex_link_edges_stamp = other._vertex_link_edges_stamp;
  _vertex_link_point_count = other._vertex_link_point_count;
  _tree_build_count = other._tree_build_count;
  _edge_membership_build_count = other._edge_membership_build_count;
  _vertex_link_build_count = other._vertex_link_build_count;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::edges_changed() -> void {
  ++_edges_generation;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::points_changed() -> void {
  ++_points_generation;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::edges_generation() const
    -> std::uint64_t {
  return _edges_generation;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::points_generation() const
    -> std::uint64_t {
  return _points_generation;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::assert_is_current(
    const geometry_type &geometry) const -> void {
  static_cast<void>(geometry);
  assert(geometry.edges_stamp == _edges_generation &&
         geometry.points_stamp == _points_generation &&
         "trueform cache: an edge mesh assembled before a stated change is "
         "reading through arrays this cache no longer answers for. Assemble "
         "it again after points_changed() or edges_changed()");
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::ensure_indices(
    const geometry_type &geometry) -> void {
  const auto point_count = static_cast<int>(geometry.number_of_points());
  if (_indices_edges_stamp == geometry.edges_stamp &&
      _indices_point_count == point_count)
    return;
  detail::fill_guard guard(_filling);
  carrier::require_point_indices(geometry.indices, point_count,
                                 "edge mesh edges");
  _indices_edges_stamp = geometry.edges_stamp;
  _indices_point_count = point_count;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::require_indices(
    const geometry_type &geometry) -> void {
  assert_is_current(geometry);
  ensure_indices(geometry);
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::ensure_tree(
    const geometry_type &geometry) -> void {
  ensure_indices(geometry);
  if (is_tree_fresh(geometry))
    return;
  detail::fill_guard guard(_filling);
  _tree = std::make_shared<tf::aabb_tree<Index, Real, Dims>>(
      tf::make_segments(geometry.edges(), geometry.points()),
      tf::config_tree(4, 4));
  _tree_edges_stamp = geometry.edges_stamp;
  _tree_points_stamp = geometry.points_stamp;
  ++_tree_build_count;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::ensure_edge_membership(
    const geometry_type &geometry) -> void {
  ensure_indices(geometry);
  if (is_edge_membership_fresh(geometry))
    return;
  detail::fill_guard guard(_filling);
  tf::edge_membership<Index> membership;
  membership.build(geometry.edges(), geometry.number_of_points());
  _edge_membership = offset_blocked_buffer<Index, Index>::from_buffer(std::move(
      static_cast<tf::offset_block_buffer<Index, Index> &>(membership)));
  _edge_membership_edges_stamp = geometry.edges_stamp;
  _edge_membership_point_count = static_cast<int>(geometry.number_of_points());
  ++_edge_membership_build_count;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::ensure_vertex_link(
    const geometry_type &geometry) -> void {
  ensure_indices(geometry);
  if (is_vertex_link_fresh(geometry))
    return;
  detail::fill_guard guard(_filling);
  tf::vertex_link<Index> link;
  link.build(geometry.edges(), geometry.number_of_points());
  _vertex_link = offset_blocked_buffer<Index, Index>::from_buffer(
      std::move(static_cast<tf::offset_block_buffer<Index, Index> &>(link)));
  _vertex_link_edges_stamp = geometry.edges_stamp;
  _vertex_link_point_count = static_cast<int>(geometry.number_of_points());
  ++_vertex_link_build_count;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::tree_handle(
    const geometry_type &geometry)
    -> std::shared_ptr<tf::aabb_tree<Index, Real, Dims>> {
  assert_is_current(geometry);
  ensure_tree(geometry);
  return _tree;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::edge_membership_handle(
    const geometry_type &geometry) -> offset_blocked_buffer<Index, Index> {
  assert_is_current(geometry);
  ensure_edge_membership(geometry);
  return _edge_membership;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::vertex_link_handle(
    const geometry_type &geometry) -> offset_blocked_buffer<Index, Index> {
  assert_is_current(geometry);
  ensure_vertex_link(geometry);
  return _vertex_link;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::set_edge_membership(
    offset_blocked_buffer<Index, Index> edge_membership,
    const geometry_type &geometry) -> void {
  assert_is_current(geometry);
  carrier::require_offset_blocks(
      edge_membership, static_cast<int>(geometry.number_of_points()),
      static_cast<int>(geometry.number_of_edges()), "edge membership");
  detail::fill_guard guard(_filling);
  _edge_membership = std::move(edge_membership);
  _edge_membership_edges_stamp = geometry.edges_stamp;
  _edge_membership_point_count = static_cast<int>(geometry.number_of_points());
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::set_vertex_link(
    offset_blocked_buffer<Index, Index> vertex_link,
    const geometry_type &geometry) -> void {
  assert_is_current(geometry);
  carrier::require_offset_blocks(
      vertex_link, static_cast<int>(geometry.number_of_points()),
      static_cast<int>(geometry.number_of_points()), "vertex link");
  detail::fill_guard guard(_filling);
  _vertex_link = std::move(vertex_link);
  _vertex_link_edges_stamp = geometry.edges_stamp;
  _vertex_link_point_count = static_cast<int>(geometry.number_of_points());
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::is_tree_built() const -> bool {
  return static_cast<bool>(_tree);
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::is_tree_fresh(
    const geometry_type &geometry) const -> bool {
  return _tree && _tree_edges_stamp == geometry.edges_stamp &&
         _tree_points_stamp == geometry.points_stamp;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::is_edge_membership_built() const
    -> bool {
  return _edge_membership.is_valid();
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::is_edge_membership_fresh(
    const geometry_type &geometry) const -> bool {
  return _edge_membership.is_valid() &&
         _edge_membership_edges_stamp == geometry.edges_stamp &&
         _edge_membership_point_count ==
             static_cast<int>(geometry.number_of_points());
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::is_vertex_link_built() const -> bool {
  return _vertex_link.is_valid();
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::is_vertex_link_fresh(
    const geometry_type &geometry) const -> bool {
  return _vertex_link.is_valid() &&
         _vertex_link_edges_stamp == geometry.edges_stamp &&
         _vertex_link_point_count ==
             static_cast<int>(geometry.number_of_points());
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::tree_build_count() const
    -> std::uint64_t {
  return _tree_build_count;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::edge_membership_build_count() const
    -> std::uint64_t {
  return _edge_membership_build_count;
}

template <typename Index, typename Real, std::size_t Dims>
auto edge_mesh_cache<Index, Real, Dims>::vertex_link_build_count() const
    -> std::uint64_t {
  return _vertex_link_build_count;
}

} // namespace tf::cpp
