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

/// @brief WASM-resident triangle mesh with lazy topology structures.
///
/// Holds flat face indices [F*3] and point coordinates [V*3] as wasm_ndarrays.
/// Topology structures are built lazily on first access and cached as wasm
/// types (wasm_ndarray / wasm_offset_blocked_buffer). Generation counters track
/// when face/point data changes to invalidate stale caches.
///
/// shared_view() creates a cheap copy that shares all data and caches via
/// shared_ptr. Each copy tracks staleness independently.
///
/// Every accessor returns a copy of the cached wasm type — safe against
/// manual .delete() on the mesh or other handles.
class wasm_mesh {
  wasm_ndarray<int> _faces;          // [F, 3]
  wasm_ndarray<float> _points;       // [V, 3]
  wasm_ndarray<float> _transformation; // [4, 4] — empty when no transform

  // Topology cache (stored as wasm types, not C++ structs)
  wasm_offset_blocked_buffer<int, int> _fm;
  wasm_ndarray<int> _mel; // [F, 3]
  wasm_offset_blocked_buffer<int, int> _fl;
  wasm_offset_blocked_buffer<int, int> _vl;

  // Normals cache
  wasm_ndarray<float> _normals;        // [F, 3]
  wasm_ndarray<float> _point_normals;  // [V, 3]

  // Half-edge cache
  std::shared_ptr<tf::half_edges<int>> _he;
  uint32_t _he_gen = 0;

  // Spatial tree cache
  std::shared_ptr<tf::aabb_tree<int, float, 3>> _tree;
  uint32_t _tree_faces_gen = 0;
  uint32_t _tree_points_gen = 0;

  // Generation counters (per-instance)
  uint32_t _faces_gen = 1;  // starts at 1 so caches are initially stale
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
  wasm_mesh() = default;

  static auto create(wasm_ndarray<int> faces, wasm_ndarray<float> points)
      -> wasm_mesh {
    wasm_mesh m;
    m._faces = std::move(faces);
    m._points = std::move(points);
    return m;
  }

  /// Create from C++ buffers (moves ownership).
  static auto from_buffers(tf::buffer<int> &&faces, tf::buffer<float> &&points)
      -> wasm_mesh {
    wasm_mesh m;
    m.assign_faces(std::move(faces));
    m.assign_points(std::move(points));
    return m;
  }

  /// Create from a triangular polygons_buffer (moves ownership).
  static auto from_polygons_buffer(tf::polygons_buffer<int, float, 3, 3> &&poly)
      -> wasm_mesh {
    return from_buffers(std::move(poly.faces_buffer().data_buffer()),
                        std::move(poly.points_buffer().data_buffer()));
  }

  // -- Data access (returns copy with shared ownership) --

  auto faces() const -> wasm_ndarray<int> { return _faces; }
  auto points() const -> wasm_ndarray<float> { return _points; }

  auto number_of_faces() const -> int {
    return _faces.is_valid() ? _faces.raw_shape()[0] : 0;
  }
  auto number_of_points() const -> int {
    return _points.is_valid() ? _points.raw_shape()[0] : 0;
  }

  // -- Data mutation --

  void set_faces(wasm_ndarray<int> faces) {
    _faces = std::move(faces);
    ++_faces_gen;
  }

  void set_points(wasm_ndarray<float> points) {
    _points = std::move(points);
    ++_points_gen;
  }

  // -- Internal: assign from C++ buffers --

  void assign_faces(tf::buffer<int> &&buf) {
    auto len = buf.size();
    _faces = wasm_ndarray<int>::from_buffer(std::move(buf),
                                          {static_cast<int>(len / 3), 3});
    ++_faces_gen;
  }

  void assign_points(tf::buffer<float> &&buf) {
    auto len = buf.size();
    _points = wasm_ndarray<float>::from_buffer(std::move(buf),
                                             {static_cast<int>(len / 3), 3});
    ++_points_gen;
  }

  // -- Transformation (per-instance, not shared) --

  auto has_transformation() const -> bool { return _transformation.is_valid(); }

  auto transformation() const -> wasm_ndarray<float> { return _transformation; }

  auto transformation_view() const -> tf::transformation_view<float, 3> {
    return tf::make_transformation_view<3>(
        const_cast<float *>(_transformation.raw_data()));
  }

  void set_transformation(const wasm_ndarray<float> &t) { _transformation = t; }
  void clear_transformation() { _transformation.destroy(); }

  // -- Shared view (copy; shares buffers via shared_ptr, no transformation) --

  auto shared_view() const -> wasm_mesh {
    wasm_mesh v = *this;
    v._transformation = wasm_ndarray<float>{};
    return v;
  }

  // -- Spatial tree (lazy build, cached) --

  void ensure_tree();

  auto tree() -> const tf::aabb_tree<int, float, 3> & {
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

  void set_face_membership(wasm_offset_blocked_buffer<int, int> fm) {
    _fm = std::move(fm);
    _fm_gen = _faces_gen;
  }
  void set_vertex_link(wasm_offset_blocked_buffer<int, int> vl) {
    _vl = std::move(vl);
    _vl_gen = _faces_gen;
  }
  void set_face_link(wasm_offset_blocked_buffer<int, int> fl) {
    _fl = std::move(fl);
    _fl_gen = _faces_gen;
  }
  void set_manifold_edge_link(wasm_ndarray<int> mel) {
    _mel = std::move(mel);
    _mel_gen = _faces_gen;
  }

  // -- Normals access (lazy build, returns copy of cached wasm type) --

  auto normals() -> wasm_ndarray<float>;
  auto point_normals() -> wasm_ndarray<float>;

  // -- Normals setters (bypass lazy build, mark as fresh) --

  void set_normals(wasm_ndarray<float> n) {
    _normals = std::move(n);
    _normals_faces_gen = _faces_gen;
    _normals_points_gen = _points_gen;
  }
  void set_point_normals(wasm_ndarray<float> pn) {
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

  // -- Spatial form construction (must be after polygons_range) --

  /// Build form from polygons + tree [+ transformation], call fn with it.
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
  }

  auto is_valid() const -> bool { return _faces.is_valid(); }

private:
  void ensure_half_edges();
  void ensure_face_membership();
  void ensure_manifold_edge_link();
  void ensure_face_link();
  void ensure_vertex_link();
  void ensure_normals();
  void ensure_point_normals();
};

} // namespace ts
} // namespace tf
