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

#include "trueform/cpp/core/cache.hpp"

#include "./carrier_arrays.hpp"
#include "./require_point_indices.hpp"

#include "trueform/core/algorithm/parallel_contains.hpp"
#include "trueform/core/algorithm/parallel_transform.hpp"
#include "trueform/core/buffer.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/policy/normals.hpp"
#include "trueform/core/polygons.hpp"
#include "trueform/core/range.hpp"
#include "trueform/core/static_size.hpp"
#include "trueform/core/unit_vectors.hpp"
#include "trueform/core/views/slide_range.hpp"
#include "trueform/core/views/zip.hpp"
#include "trueform/cpp/core/detail/fill_guard.hpp"
#include "trueform/geometry/compute_normals.hpp"
#include "trueform/geometry/compute_point_normals.hpp"
#include "trueform/spatial/tree_config.hpp"
#include "trueform/topology/face_link.hpp"
#include "trueform/topology/face_membership.hpp"
#include "trueform/topology/face_membership_like.hpp"
#include "trueform/topology/manifold_edge_link.hpp"
#include "trueform/topology/policy/face_membership.hpp"
#include "trueform/topology/vertex_link.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace tf::cpp {
namespace cache_detail {

/// Core's own polygons carry no invariant of their own, so a mixed mesh's raw
/// blocks are read here: they begin at zero and span the packed indices. A
/// reading that states no offsets at all states no faces, which is what an
/// empty carrier looks like wherever its memory came from.
template <typename Index>
auto require_face_span(
    const tf::range<const Index *, tf::dynamic_size> &offsets,
    std::size_t index_count) -> void {
  if (offsets.size() == 0)
    return;
  if (offsets[0] != Index{0})
    throw std::invalid_argument("mesh faces: offsets must begin with zero");
  if (offsets[offsets.size() - 1] != static_cast<Index>(index_count))
    throw std::invalid_argument(
        "mesh faces: offsets must span the packed indices");
}

/// A face is at least three points — the mesh's own fact, which nondecreasing
/// offsets follow from.
template <typename Index>
auto require_face_arity(
    const tf::range<const Index *, tf::dynamic_size> &offsets) -> void {
  if (offsets.size() < 2)
    return;
  const auto refused = tf::parallel_contains(
      tf::make_slide_range<2>(offsets),
      [](const auto &block) { return block[1] - block[0] < Index{3}; },
      tf::checked);
  if (refused)
    throw std::invalid_argument(
        "mesh faces: a face requires at least three indices");
}

/// The coordinates a reading stands on are whole points.
template <std::size_t Dims>
auto require_point_span(std::size_t coordinate_count) -> void {
  if (coordinate_count % Dims != 0)
    throw std::invalid_argument(
        "mesh points: storage is not dimension-aligned");
}

/// A per-face-side fact of a mixed reading has one value per side, so its
/// blocks ARE the reading's faces: same count, same spans. The reading owns
/// those spans; this compares a stated structure against them rather than
/// deriving them a second time.
template <typename Index, typename Offsets>
auto require_face_aligned_blocks(
    const Offsets &offsets,
    const tf::range<const Index *, tf::dynamic_size> &face_offsets,
    const char *name) -> void {
  if (offsets.size() != face_offsets.size())
    throw std::invalid_argument(std::string(name) +
                                " must state one block per face");
  const auto misaligned = tf::parallel_contains(
      tf::zip(offsets, face_offsets),
      [](const auto &pair) { return std::get<0>(pair) != std::get<1>(pair); },
      tf::checked);
  if (misaligned)
    throw std::invalid_argument(std::string(name) +
                                " blocks must span the faces of this reading");
}

} // namespace cache_detail

/// A copy shares nothing, so what it carries is what it can own outright — the
/// four flat structures, deep. The tree and the half edges hold views into
/// their own storage, and the winding moments mirror the tree's nodes, so a
/// copy builds them when it is asked for them. It built none of what it
/// carries, so every build count starts at zero.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
cache<Index, Real, Dims, Ngon>::cache(const cache &other) {
  copy_from(other);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
cache<Index, Real, Dims, Ngon>::cache(cache &&other) noexcept {
  move_from(std::move(other));
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::operator=(const cache &other) -> cache & {
  if (this != &other)
    copy_from(other);
  return *this;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::operator=(cache &&other) noexcept
    -> cache & {
  if (this != &other)
    move_from(std::move(other));
  return *this;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::copy_from(const cache &other) -> void {
  _face_membership = other._face_membership.deep_copy();
  _manifold_edge_link = other._manifold_edge_link.deep_copy();
  _face_link = other._face_link.deep_copy();
  _vertex_link = other._vertex_link.deep_copy();
  _face_normals = other._face_normals.deep_copy();
  _point_normals = other._point_normals.deep_copy();
  _half_edges.reset();
  _tree.reset();
  _winding_moments.reset();
  _faces_generation = other._faces_generation;
  _points_generation = other._points_generation;
  _indices_faces_stamp = other._indices_faces_stamp;
  _indices_point_count = other._indices_point_count;
  _tree_faces_stamp = 0;
  _tree_points_stamp = 0;
  _winding_moments_tree_build = 0;
  _half_edges_faces_stamp = 0;
  _half_edges_point_count = -1;
  _face_membership_faces_stamp = other._face_membership_faces_stamp;
  _face_membership_point_count = other._face_membership_point_count;
  _manifold_edge_link_faces_stamp = other._manifold_edge_link_faces_stamp;
  _face_link_faces_stamp = other._face_link_faces_stamp;
  _vertex_link_faces_stamp = other._vertex_link_faces_stamp;
  _vertex_link_point_count = other._vertex_link_point_count;
  _face_normals_faces_stamp = other._face_normals_faces_stamp;
  _face_normals_points_stamp = other._face_normals_points_stamp;
  _point_normals_faces_stamp = other._point_normals_faces_stamp;
  _point_normals_points_stamp = other._point_normals_points_stamp;
  _tree_build_count = 0;
  _winding_moments_build_count = 0;
  _half_edges_build_count = 0;
  _face_membership_build_count = 0;
  _manifold_edge_link_build_count = 0;
  _face_link_build_count = 0;
  _vertex_link_build_count = 0;
  _face_normals_build_count = 0;
  _point_normals_build_count = 0;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::move_from(cache &&other) -> void {
  _face_membership = std::move(other._face_membership);
  _manifold_edge_link = std::move(other._manifold_edge_link);
  _face_link = std::move(other._face_link);
  _vertex_link = std::move(other._vertex_link);
  _face_normals = std::move(other._face_normals);
  _point_normals = std::move(other._point_normals);
  _half_edges = std::move(other._half_edges);
  _tree = std::move(other._tree);
  _winding_moments = std::move(other._winding_moments);
  _faces_generation = other._faces_generation;
  _points_generation = other._points_generation;
  _indices_faces_stamp = other._indices_faces_stamp;
  _indices_point_count = other._indices_point_count;
  _tree_faces_stamp = other._tree_faces_stamp;
  _tree_points_stamp = other._tree_points_stamp;
  _winding_moments_tree_build = other._winding_moments_tree_build;
  _half_edges_faces_stamp = other._half_edges_faces_stamp;
  _half_edges_point_count = other._half_edges_point_count;
  _face_membership_faces_stamp = other._face_membership_faces_stamp;
  _face_membership_point_count = other._face_membership_point_count;
  _manifold_edge_link_faces_stamp = other._manifold_edge_link_faces_stamp;
  _face_link_faces_stamp = other._face_link_faces_stamp;
  _vertex_link_faces_stamp = other._vertex_link_faces_stamp;
  _vertex_link_point_count = other._vertex_link_point_count;
  _face_normals_faces_stamp = other._face_normals_faces_stamp;
  _face_normals_points_stamp = other._face_normals_points_stamp;
  _point_normals_faces_stamp = other._point_normals_faces_stamp;
  _point_normals_points_stamp = other._point_normals_points_stamp;
  _tree_build_count = other._tree_build_count;
  _winding_moments_build_count = other._winding_moments_build_count;
  _half_edges_build_count = other._half_edges_build_count;
  _face_membership_build_count = other._face_membership_build_count;
  _manifold_edge_link_build_count = other._manifold_edge_link_build_count;
  _face_link_build_count = other._face_link_build_count;
  _vertex_link_build_count = other._vertex_link_build_count;
  _face_normals_build_count = other._face_normals_build_count;
  _point_normals_build_count = other._point_normals_build_count;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::faces_changed() -> void {
  ++_faces_generation;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::points_changed() -> void {
  ++_points_generation;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::faces_generation() const -> std::uint64_t {
  return _faces_generation;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::points_generation() const
    -> std::uint64_t {
  return _points_generation;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::assert_is_current(
    const geometry_type &geometry) const -> void {
  static_cast<void>(geometry);
  assert(geometry.faces_stamp == _faces_generation &&
         geometry.points_stamp == _points_generation &&
         "trueform cache: a mesh assembled before a stated change is reading "
         "through arrays this cache no longer answers for. Assemble the mesh "
         "again after points_changed() or faces_changed()");
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::ensure_indices(
    const geometry_type &geometry) -> void {
  const auto point_count = static_cast<int>(geometry.number_of_points());
  if (_indices_faces_stamp == geometry.faces_stamp &&
      _indices_point_count == point_count)
    return;
  detail::fill_guard guard(_filling);
  cache_detail::require_point_span<Dims>(geometry.coordinates.size());
  if constexpr (Ngon == tf::dynamic_size) {
    cache_detail::require_face_span<Index>(geometry.offsets,
                                           geometry.indices.size());
    cache_detail::require_face_arity<Index>(geometry.offsets);
  }
  carrier::require_point_indices(geometry.indices, point_count, "mesh faces");
  _indices_faces_stamp = geometry.faces_stamp;
  _indices_point_count = point_count;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::require_indices(
    const geometry_type &geometry) -> void {
  assert_is_current(geometry);
  ensure_indices(geometry);
}

/// EVERY FILL HAS ONE SHAPE, and it is the order two different facts have to be
/// asked in: the DOOR first (`ensure_indices`, which refuses a reading whose
/// corners name points it does not have), then this structure's own freshness,
/// then what it stands on, then the guard, then the build. The door cannot ride
/// behind the freshness check, because a structure keyed on the faces stamp
/// alone — the manifold edge link, the face link — is fresh after a points
/// shrink and would answer past a door that never ran.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::ensure_tree(const geometry_type &geometry)
    -> void {
  ensure_indices(geometry);
  if (is_tree_fresh(geometry))
    return;
  detail::fill_guard guard(_filling);
  _tree = std::make_shared<tf::aabb_tree<Index, Real, Dims>>(
      tf::make_polygons(geometry.faces(), geometry.points()),
      tf::config_tree(4, 12));
  _tree_faces_stamp = geometry.faces_stamp;
  _tree_points_stamp = geometry.points_stamp;
  ++_tree_build_count;
}

/// A moment row is meaningful only beside the node row it mirrors, so the
/// tree's BUILD is what these are fresh for: a tree that was built again
/// carries different nodes whatever the generations that asked for it were.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::ensure_winding_moments(
    const geometry_type &geometry) -> void {
  ensure_indices(geometry);
  if (is_winding_moments_fresh(geometry))
    return;
  if constexpr (Dims == 3) {
    ensure_tree(geometry);
    detail::fill_guard guard(_filling);
    auto moments = std::make_shared<tf::winding_moments<Real>>();
    moments->build(*_tree,
                   tf::make_polygons(geometry.faces(), geometry.points()));
    _winding_moments = std::move(moments);
    _winding_moments_tree_build = _tree_build_count;
    ++_winding_moments_build_count;
  } else {
    throw std::invalid_argument("winding moments require a 3D mesh");
  }
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::ensure_face_membership(
    const geometry_type &geometry)
    -> const offset_blocked_buffer<Index, Index> & {
  ensure_indices(geometry);
  if (is_face_membership_fresh(geometry))
    return _face_membership;
  detail::fill_guard guard(_filling);
  tf::face_membership<Index> face_membership;
  face_membership.build(tf::make_polygons(geometry.faces(), geometry.points()));
  _face_membership = offset_blocked_buffer<Index, Index>::from_buffer(std::move(
      static_cast<tf::offset_block_buffer<Index, Index> &>(face_membership)));
  _face_membership_faces_stamp = geometry.faces_stamp;
  _face_membership_point_count = static_cast<int>(geometry.number_of_points());
  ++_face_membership_build_count;
  return _face_membership;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::ensure_half_edges(
    const geometry_type &geometry) -> void {
  ensure_indices(geometry);
  if (is_half_edges_fresh(geometry))
    return;
  ensure_face_membership(geometry);
  detail::fill_guard guard(_filling);
  auto half_edges = std::make_shared<tf::half_edges<Index>>();
  half_edges->build(geometry.faces(), tf::make_face_membership_like(
                                          _face_membership.make_range()));
  _half_edges = std::move(half_edges);
  _half_edges_faces_stamp = geometry.faces_stamp;
  _half_edges_point_count = static_cast<int>(geometry.number_of_points());
  ++_half_edges_build_count;
}

/// One peer per face side, in the shape that arity states a per-face-side fact
/// in: a triangle reading's stride is three, a mixed reading carries its own
/// face-aligned offsets. It is the shape the free entry produces and the shape
/// the door takes, so one fact has one shape however it was filled.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::ensure_manifold_edge_link(
    const geometry_type &geometry) -> void {
  ensure_indices(geometry);
  if (is_manifold_edge_link_fresh(geometry))
    return;
  ensure_face_membership(geometry);
  detail::fill_guard guard(_filling);
  tf::manifold_edge_link<Index, Ngon> peers;
  peers.build(geometry.faces(),
              tf::make_face_membership_like(_face_membership.make_range()));
  auto &source = peers.data_buffer();
  tf::buffer<Index> output;
  output.allocate(source.size());
  tf::parallel_transform(
      source, output, [](const auto &peer) { return peer.face_peer; },
      tf::checked);
  if constexpr (Ngon == 3) {
    const auto count = static_cast<int>(output.size());
    _manifold_edge_link =
        nd_array<Index>::from_buffer(std::move(output), {count / 3, 3});
  } else {
    tf::offset_block_buffer<Index, Index> blocks;
    blocks.offsets_buffer() = std::move(peers.offsets_buffer());
    blocks.data_buffer() = std::move(output);
    _manifold_edge_link =
        offset_blocked_buffer<Index, Index>::from_buffer(std::move(blocks));
  }
  _manifold_edge_link_faces_stamp = geometry.faces_stamp;
  ++_manifold_edge_link_build_count;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::ensure_face_link(
    const geometry_type &geometry) -> void {
  ensure_indices(geometry);
  if (is_face_link_fresh(geometry))
    return;
  ensure_face_membership(geometry);
  detail::fill_guard guard(_filling);
  tf::face_link<Index> face_link;
  face_link.build(geometry.faces(),
                  tf::make_face_membership_like(_face_membership.make_range()));
  _face_link = offset_blocked_buffer<Index, Index>::from_buffer(std::move(
      static_cast<tf::offset_block_buffer<Index, Index> &>(face_link)));
  _face_link_faces_stamp = geometry.faces_stamp;
  ++_face_link_build_count;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::ensure_vertex_link(
    const geometry_type &geometry) -> void {
  ensure_indices(geometry);
  if (is_vertex_link_fresh(geometry))
    return;
  ensure_face_membership(geometry);
  detail::fill_guard guard(_filling);
  tf::vertex_link<Index> vertex_link;
  vertex_link.build(geometry.faces(), tf::make_face_membership_like(
                                          _face_membership.make_range()));
  _vertex_link = offset_blocked_buffer<Index, Index>::from_buffer(std::move(
      static_cast<tf::offset_block_buffer<Index, Index> &>(vertex_link)));
  _vertex_link_faces_stamp = geometry.faces_stamp;
  _vertex_link_point_count = static_cast<int>(geometry.number_of_points());
  ++_vertex_link_build_count;
}

/// A normal is of a three-dimensional surface by its own definition, so a
/// carrier of any other dimension has none to remember — and the entries that
/// ask refuse by substitution, so the throw states a case the surface cannot
/// reach.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::ensure_face_normals(
    const geometry_type &geometry) -> void {
  ensure_indices(geometry);
  if (is_face_normals_fresh(geometry))
    return;
  if constexpr (Dims == 3) {
    detail::fill_guard guard(_filling);
    auto normals = tf::compute_normals(
        tf::make_polygons(geometry.faces(), geometry.points()));
    const auto count = static_cast<int>(normals.size());
    _face_normals = nd_array<Real>::from_buffer(
        std::move(normals.data_buffer()), {count, 3});
    _face_normals_faces_stamp = geometry.faces_stamp;
    _face_normals_points_stamp = geometry.points_stamp;
    ++_face_normals_build_count;
  } else {
    throw std::invalid_argument("face normals require a 3D mesh");
  }
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::ensure_point_normals(
    const geometry_type &geometry) -> void {
  ensure_indices(geometry);
  if (is_point_normals_fresh(geometry))
    return;
  if constexpr (Dims == 3) {
    ensure_face_normals(geometry);
    ensure_face_membership(geometry);
    detail::fill_guard guard(_filling);
    auto normals = tf::compute_point_normals(
        tf::make_polygons(geometry.faces(), geometry.points()) |
        tf::tag(tf::make_face_membership_like(_face_membership.make_range())) |
        tf::tag_normals(tf::make_unit_vectors<3>(
            std::as_const(_face_normals).make_range())));
    const auto count = static_cast<int>(normals.size());
    _point_normals = nd_array<Real>::from_buffer(
        std::move(normals.data_buffer()), {count, 3});
    _point_normals_faces_stamp = geometry.faces_stamp;
    _point_normals_points_stamp = geometry.points_stamp;
    ++_point_normals_build_count;
  } else {
    throw std::invalid_argument("point normals require a 3D mesh");
  }
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::tree_handle(const geometry_type &geometry)
    -> std::shared_ptr<tf::aabb_tree<Index, Real, Dims>> {
  assert_is_current(geometry);
  ensure_tree(geometry);
  return _tree;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::winding_moments_handle(
    const geometry_type &geometry)
    -> std::shared_ptr<tf::winding_moments<Real>> {
  assert_is_current(geometry);
  ensure_winding_moments(geometry);
  return _winding_moments;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::half_edges_handle(
    const geometry_type &geometry) -> std::shared_ptr<tf::half_edges<Index>> {
  assert_is_current(geometry);
  ensure_half_edges(geometry);
  return _half_edges;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::face_membership_handle(
    const geometry_type &geometry) -> offset_blocked_buffer<Index, Index> {
  assert_is_current(geometry);
  return ensure_face_membership(geometry);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::manifold_edge_link_handle(
    const geometry_type &geometry) -> face_blocks_type {
  assert_is_current(geometry);
  ensure_manifold_edge_link(geometry);
  return _manifold_edge_link;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::face_link_handle(
    const geometry_type &geometry) -> offset_blocked_buffer<Index, Index> {
  assert_is_current(geometry);
  ensure_face_link(geometry);
  return _face_link;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::vertex_link_handle(
    const geometry_type &geometry) -> offset_blocked_buffer<Index, Index> {
  assert_is_current(geometry);
  ensure_vertex_link(geometry);
  return _vertex_link;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::face_normals_handle(
    const geometry_type &geometry) -> nd_array<Real> {
  assert_is_current(geometry);
  ensure_face_normals(geometry);
  return _face_normals;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::point_normals_handle(
    const geometry_type &geometry) -> nd_array<Real> {
  assert_is_current(geometry);
  ensure_point_normals(geometry);
  return _point_normals;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::set_half_edges(
    tf::half_edges<Index> &&half_edges, const geometry_type &geometry) -> void {
  assert_is_current(geometry);
  if (half_edges.n_faces() != static_cast<Index>(geometry.number_of_faces()) ||
      half_edges.n_vertices() !=
          static_cast<Index>(geometry.number_of_points()))
    throw std::invalid_argument(
        "half edges must match the face and point counts");
  detail::fill_guard guard(_filling);
  _half_edges = std::make_shared<tf::half_edges<Index>>(std::move(half_edges));
  _half_edges_faces_stamp = geometry.faces_stamp;
  _half_edges_point_count = static_cast<int>(geometry.number_of_points());
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::set_face_membership(
    offset_blocked_buffer<Index, Index> face_membership,
    const geometry_type &geometry) -> void {
  assert_is_current(geometry);
  carrier::require_offset_blocks(
      face_membership, static_cast<int>(geometry.number_of_points()),
      static_cast<int>(geometry.number_of_faces()), "face membership");
  detail::fill_guard guard(_filling);
  _face_membership = std::move(face_membership);
  _face_membership_faces_stamp = geometry.faces_stamp;
  _face_membership_point_count = static_cast<int>(geometry.number_of_points());
}

/// A peer is a face id, so the door reads the VALUES as well as the shape: the
/// one structure whose contents can name a face this reading does not have.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::set_manifold_edge_link(
    face_blocks_type manifold_edge_link, const geometry_type &geometry)
    -> void {
  assert_is_current(geometry);
  const auto face_count = static_cast<int>(geometry.number_of_faces());
  if constexpr (Ngon == 3) {
    carrier::require_columns<3>(manifold_edge_link, "manifold edge link");
    if (manifold_edge_link.shape_at(0) != face_count)
      throw std::invalid_argument(
          "manifold edge link must match the face count");
    carrier::require_face_peers<Index>(manifold_edge_link.make_range(),
                                       face_count, "manifold edge link");
  } else {
    static_cast<void>(carrier::require_offset_blocks(
        manifold_edge_link, face_count, carrier::any_count,
        "manifold edge link"));
    cache_detail::require_face_aligned_blocks<Index>(
        manifold_edge_link.offsets().make_range(), geometry.offsets,
        "manifold edge link");
    carrier::require_face_peers<Index>(manifold_edge_link.data().make_range(),
                                       face_count, "manifold edge link");
  }
  detail::fill_guard guard(_filling);
  _manifold_edge_link = std::move(manifold_edge_link);
  _manifold_edge_link_faces_stamp = geometry.faces_stamp;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::set_face_link(
    offset_blocked_buffer<Index, Index> face_link,
    const geometry_type &geometry) -> void {
  assert_is_current(geometry);
  carrier::require_offset_blocks(
      face_link, static_cast<int>(geometry.number_of_faces()),
      static_cast<int>(geometry.number_of_faces()), "face link");
  detail::fill_guard guard(_filling);
  _face_link = std::move(face_link);
  _face_link_faces_stamp = geometry.faces_stamp;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::set_vertex_link(
    offset_blocked_buffer<Index, Index> vertex_link,
    const geometry_type &geometry) -> void {
  assert_is_current(geometry);
  carrier::require_offset_blocks(
      vertex_link, static_cast<int>(geometry.number_of_points()),
      static_cast<int>(geometry.number_of_points()), "vertex link");
  detail::fill_guard guard(_filling);
  _vertex_link = std::move(vertex_link);
  _vertex_link_faces_stamp = geometry.faces_stamp;
  _vertex_link_point_count = static_cast<int>(geometry.number_of_points());
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_tree_built() const -> bool {
  return static_cast<bool>(_tree);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_tree_fresh(
    const geometry_type &geometry) const -> bool {
  return _tree && _tree_faces_stamp == geometry.faces_stamp &&
         _tree_points_stamp == geometry.points_stamp;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_winding_moments_built() const -> bool {
  return static_cast<bool>(_winding_moments);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_winding_moments_fresh(
    const geometry_type &geometry) const -> bool {
  return _winding_moments && is_tree_fresh(geometry) &&
         _winding_moments_tree_build == _tree_build_count;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_half_edges_built() const -> bool {
  return static_cast<bool>(_half_edges);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_half_edges_fresh(
    const geometry_type &geometry) const -> bool {
  return _half_edges && _half_edges_faces_stamp == geometry.faces_stamp &&
         _half_edges_point_count ==
             static_cast<int>(geometry.number_of_points());
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_face_membership_built() const -> bool {
  return _face_membership.is_valid();
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_face_membership_fresh(
    const geometry_type &geometry) const -> bool {
  return _face_membership.is_valid() &&
         _face_membership_faces_stamp == geometry.faces_stamp &&
         _face_membership_point_count ==
             static_cast<int>(geometry.number_of_points());
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_manifold_edge_link_built() const
    -> bool {
  return _manifold_edge_link.is_valid();
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_manifold_edge_link_fresh(
    const geometry_type &geometry) const -> bool {
  return _manifold_edge_link.is_valid() &&
         _manifold_edge_link_faces_stamp == geometry.faces_stamp;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_face_link_built() const -> bool {
  return _face_link.is_valid();
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_face_link_fresh(
    const geometry_type &geometry) const -> bool {
  return _face_link.is_valid() &&
         _face_link_faces_stamp == geometry.faces_stamp;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_vertex_link_built() const -> bool {
  return _vertex_link.is_valid();
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_vertex_link_fresh(
    const geometry_type &geometry) const -> bool {
  return _vertex_link.is_valid() &&
         _vertex_link_faces_stamp == geometry.faces_stamp &&
         _vertex_link_point_count ==
             static_cast<int>(geometry.number_of_points());
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_face_normals_built() const -> bool {
  return _face_normals.is_valid();
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_face_normals_fresh(
    const geometry_type &geometry) const -> bool {
  return _face_normals.is_valid() &&
         _face_normals_faces_stamp == geometry.faces_stamp &&
         _face_normals_points_stamp == geometry.points_stamp;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_point_normals_built() const -> bool {
  return _point_normals.is_valid();
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::is_point_normals_fresh(
    const geometry_type &geometry) const -> bool {
  return _point_normals.is_valid() &&
         _point_normals_faces_stamp == geometry.faces_stamp &&
         _point_normals_points_stamp == geometry.points_stamp;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::tree_build_count() const -> std::uint64_t {
  return _tree_build_count;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::winding_moments_build_count() const
    -> std::uint64_t {
  return _winding_moments_build_count;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::half_edges_build_count() const
    -> std::uint64_t {
  return _half_edges_build_count;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::face_membership_build_count() const
    -> std::uint64_t {
  return _face_membership_build_count;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::manifold_edge_link_build_count() const
    -> std::uint64_t {
  return _manifold_edge_link_build_count;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::face_link_build_count() const
    -> std::uint64_t {
  return _face_link_build_count;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::vertex_link_build_count() const
    -> std::uint64_t {
  return _vertex_link_build_count;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::face_normals_build_count() const
    -> std::uint64_t {
  return _face_normals_build_count;
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cache<Index, Real, Dims, Ngon>::point_normals_build_count() const
    -> std::uint64_t {
  return _point_normals_build_count;
}

} // namespace tf::cpp
