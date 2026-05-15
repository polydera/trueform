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

#include "trueform/core/faces.hpp"
#include "trueform/core/points.hpp"
#include "trueform/core/polygons.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/core/views/blocked_range.hpp"
#include "trueform/core/views/mapped_range.hpp"
#include "trueform/topology/face_link_like.hpp"
#include "trueform/topology/face_membership_like.hpp"
#include "trueform/topology/manifold_edge_link_like.hpp"
#include "trueform/topology/manifold_edge_peer.hpp"
#include "trueform/topology/vertex_link_like.hpp"
#include "trueform/core/transformation_view.hpp"
#include "trueform/spatial/aabb_tree.hpp"
#include "trueform/topology/half_edges.hpp"
#include "trueform/core/policy/frame.hpp"
#include "trueform/spatial/policy/tree.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/core/wasm_offset_blocked_buffer.hpp"
#include <cstdint>
#include <memory>

namespace tf {
namespace ts {

/// @brief Inner data class for wasm_mesh (PIMPL).
///
/// Holds flat face indices [F*3] and point coordinates [V*3] as
/// wasm_ndarrays. Topology structures (face_membership, manifold_edge_link,
/// face_link, vertex_link) and normals (face + point) are built lazily on
/// first access via private ensure_* helpers; public access is via the
/// accessors which trigger the build.
///
/// Multiple wasm_mesh handles can share the same mesh_data via shared_ptr —
/// that is how cache state is shared across copies (e.g. async lambda
/// captures).
template <typename Real>
class mesh_data {
  wasm_ndarray<int> _faces;            // [F, 3]
  wasm_ndarray<Real> _points;          // [V, 3]
  wasm_ndarray<Real> _transformation;  // [4, 4] — empty when no transform

  // Topology cache (stored as wasm types, not C++ structs)
  wasm_offset_blocked_buffer<int, int> _fm;
  wasm_ndarray<int> _mel; // [F, 3]
  wasm_offset_blocked_buffer<int, int> _fl;
  wasm_offset_blocked_buffer<int, int> _vl;

  // Normals cache
  wasm_ndarray<Real> _normals;       // [F, 3]
  wasm_ndarray<Real> _point_normals; // [V, 3]

  // Half-edge cache
  std::shared_ptr<tf::half_edges<int>> _he;
  uint32_t _he_gen = 0;

  // Spatial tree cache
  std::shared_ptr<tf::aabb_tree<int, Real, 3>> _tree;
  uint32_t _tree_faces_gen = 0;
  uint32_t _tree_points_gen = 0;

  // Generation counters
  uint32_t _faces_gen = 1; // starts at 1 so caches are initially stale
  uint32_t _points_gen = 1;
  uint32_t _fm_gen = 0;
  uint32_t _mel_gen = 0;
  uint32_t _fl_gen = 0;
  uint32_t _vl_gen = 0;
  uint32_t _normals_faces_gen = 0;
  uint32_t _normals_points_gen = 0;
  uint32_t _point_normals_faces_gen = 0;
  uint32_t _point_normals_points_gen = 0;

public:
  mesh_data() = default;

  // -- Data access (returns copy with shared ownership) --

  auto faces() const -> wasm_ndarray<int> { return _faces; }
  auto points() const -> wasm_ndarray<Real> { return _points; }

  auto number_of_faces() const -> int {
    return _faces.is_valid() ? _faces.raw_shape()[0] : 0;
  }
  auto number_of_points() const -> int {
    return _points.is_valid() ? _points.raw_shape()[0] : 0;
  }

  // -- Data mutation --

  auto set_faces(wasm_ndarray<int> faces) -> void {
    _faces = std::move(faces);
    ++_faces_gen;
  }

  auto set_points(wasm_ndarray<Real> points) -> void {
    _points = std::move(points);
    ++_points_gen;
  }

  // -- Internal: assign from C++ buffers --

  auto assign_faces(tf::buffer<int> &&buf) -> void {
    auto len = buf.size();
    _faces = wasm_ndarray<int>::from_buffer(std::move(buf),
                                            {static_cast<int>(len / 3), 3});
    ++_faces_gen;
  }

  auto assign_points(tf::buffer<Real> &&buf) -> void {
    auto len = buf.size();
    _points = wasm_ndarray<Real>::from_buffer(std::move(buf),
                                              {static_cast<int>(len / 3), 3});
    ++_points_gen;
  }

  // -- Transformation --

  auto has_transformation() const -> bool { return _transformation.is_valid(); }

  auto transformation() const -> wasm_ndarray<Real> { return _transformation; }

  auto transformation_view() const -> tf::transformation_view<Real, 3> {
    return tf::make_transformation_view<3>(
        const_cast<Real *>(_transformation.raw_data()));
  }

  auto set_transformation(const wasm_ndarray<Real> &t) -> void {
    _transformation = t;
  }
  auto clear_transformation() -> void { _transformation.destroy(); }

  // -- Spatial tree (lazy build, cached) --

  auto tree() -> const tf::aabb_tree<int, Real, 3> & {
    ensure_tree();
    return *_tree;
  }

  // -- Half-edge access (lazy build, cached) --

  auto half_edges() -> tf::half_edges<int> & {
    ensure_half_edges();
    return *_he;
  }

  auto set_half_edges(tf::half_edges<int> &&he) -> void {
    _he = std::make_shared<tf::half_edges<int>>(std::move(he));
    _he_gen = _faces_gen;
  }

  // -- Topology access (lazy build, returns copy of cached wasm type) --

  auto face_membership() -> wasm_offset_blocked_buffer<int, int>;
  auto manifold_edge_link() -> wasm_ndarray<int>;
  auto face_link() -> wasm_offset_blocked_buffer<int, int>;
  auto vertex_link() -> wasm_offset_blocked_buffer<int, int>;

  // -- Topology setters (bypass lazy build, mark as fresh) --

  auto set_face_membership(wasm_offset_blocked_buffer<int, int> fm) -> void {
    _fm = std::move(fm);
    _fm_gen = _faces_gen;
  }
  auto set_vertex_link(wasm_offset_blocked_buffer<int, int> vl) -> void {
    _vl = std::move(vl);
    _vl_gen = _faces_gen;
  }
  auto set_face_link(wasm_offset_blocked_buffer<int, int> fl) -> void {
    _fl = std::move(fl);
    _fl_gen = _faces_gen;
  }
  auto set_manifold_edge_link(wasm_ndarray<int> mel) -> void {
    _mel = std::move(mel);
    _mel_gen = _faces_gen;
  }

  // -- Normals access (lazy build, returns copy of cached wasm type) --

  auto normals() -> wasm_ndarray<Real>;
  auto point_normals() -> wasm_ndarray<Real>;

  // -- Normals setters (bypass lazy build, mark as fresh) --

  auto set_normals(wasm_ndarray<Real> n) -> void {
    _normals = std::move(n);
    _normals_faces_gen = _faces_gen;
    _normals_points_gen = _points_gen;
  }
  auto set_point_normals(wasm_ndarray<Real> pn) -> void {
    _point_normals = std::move(pn);
    _point_normals_faces_gen = _faces_gen;
    _point_normals_points_gen = _points_gen;
  }

  // -- Internal: trueform range views over raw data --

  auto faces_range() const {
    return tf::make_faces<3>(_faces.make_range());
  }

  auto points_range() const {
    return tf::make_points<3>(_points.make_range());
  }

  auto polygons_range() const {
    return tf::make_polygons(faces_range(), points_range());
  }

  auto face_membership_range() {
    ensure_face_membership();
    return tf::make_face_membership_like(_fm.make_range());
  }

  auto face_link_range() {
    ensure_face_link();
    return tf::make_face_link_like(_fl.make_range());
  }

  auto vertex_link_range() {
    ensure_vertex_link();
    return tf::make_vertex_link_like(_vl.make_range());
  }

  auto manifold_edge_link_range() {
    ensure_manifold_edge_link();
    // mel stored as int [F,3]; map each int -> manifold_edge_peer<int>
    struct dref_t {
      auto operator()(int i) const -> tf::manifold_edge_peer<int> {
        return {i};
      }
    };
    auto r = tf::make_mapped_range(_mel.make_range(), dref_t{});
    return tf::make_manifold_edge_link_like(tf::make_blocked_range<3>(r));
  }

  // -- Spatial form construction --

  template <typename Fn> auto with_form(Fn &&fn) -> decltype(auto) {
    ensure_tree();
    if (has_transformation()) {
      auto form = polygons_range() | tf::tag(*_tree) |
                  tf::tag(transformation_view());
      return fn(form);
    } else {
      auto form = polygons_range() | tf::tag(*_tree);
      return fn(form);
    }
  }

  // -- Lifecycle --

  auto destroy() -> void {
    _faces.destroy();
    _points.destroy();
    _transformation.destroy();
    _fm.destroy();
    _mel.destroy();
    _fl.destroy();
    _vl.destroy();
    _normals.destroy();
    _point_normals.destroy();
    _he.reset();
    _tree.reset();
  }

  auto is_valid() const -> bool { return _faces.is_valid(); }

  // -- Cache state inspectors (diagnostic — not user-facing TS API) --
  //
  // is_X_built : the cache slot has been populated at some point (may be stale).
  // is_X_fresh : populated AND generations match the current source data.

  auto is_tree_built() const -> bool { return _tree != nullptr; }
  auto is_tree_fresh() const -> bool {
    return _tree != nullptr && _tree_faces_gen == _faces_gen &&
           _tree_points_gen == _points_gen;
  }

  auto is_half_edges_built() const -> bool { return _he != nullptr; }
  auto is_half_edges_fresh() const -> bool {
    return _he != nullptr && _he_gen == _faces_gen;
  }

  auto is_face_membership_built() const -> bool { return _fm.is_valid(); }
  auto is_face_membership_fresh() const -> bool {
    return _fm.is_valid() && _fm_gen == _faces_gen;
  }

  auto is_manifold_edge_link_built() const -> bool { return _mel.is_valid(); }
  auto is_manifold_edge_link_fresh() const -> bool {
    return _mel.is_valid() && _mel_gen == _faces_gen;
  }

  auto is_face_link_built() const -> bool { return _fl.is_valid(); }
  auto is_face_link_fresh() const -> bool {
    return _fl.is_valid() && _fl_gen == _faces_gen;
  }

  auto is_vertex_link_built() const -> bool { return _vl.is_valid(); }
  auto is_vertex_link_fresh() const -> bool {
    return _vl.is_valid() && _vl_gen == _faces_gen;
  }

  auto is_normals_built() const -> bool { return _normals.is_valid(); }
  auto is_normals_fresh() const -> bool {
    return _normals.is_valid() && _normals_faces_gen == _faces_gen &&
           _normals_points_gen == _points_gen;
  }

  auto is_point_normals_built() const -> bool { return _point_normals.is_valid(); }
  auto is_point_normals_fresh() const -> bool {
    return _point_normals.is_valid() &&
           _point_normals_faces_gen == _faces_gen &&
           _point_normals_points_gen == _points_gen;
  }

private:
  auto ensure_tree() -> void;
  auto ensure_half_edges() -> void;
  auto ensure_face_membership() -> void;
  auto ensure_manifold_edge_link() -> void;
  auto ensure_face_link() -> void;
  auto ensure_vertex_link() -> void;
  auto ensure_normals() -> void;
  auto ensure_point_normals() -> void;
};

extern template class mesh_data<float>;
extern template class mesh_data<double>;

} // namespace ts
} // namespace tf
