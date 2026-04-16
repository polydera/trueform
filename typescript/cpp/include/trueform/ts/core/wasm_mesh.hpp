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

#include "trueform/ts/core/mesh_data.hpp"
#include <memory>
#include <utility>

namespace tf {
namespace ts {

/// @brief WASM-resident triangle mesh handle.
///
/// Thin wrapper holding a std::shared_ptr<mesh_data>. All methods forward
/// to the inner data. Default copy = shared_ptr copy, so handle copies
/// share the same underlying data and any lazily built caches (tree,
/// normals, topology). This is what makes async lambda captures work:
/// `[m = m, ...]` copies the handle, the worker writes caches into the
/// shared mesh_data, and the JS-visible original sees them.
///
/// shallow_copy() forks the data (new mesh_data, member-wise copy —
/// internal storage still shared via wasm_ndarray's shared_ptr) and
/// clears the transformation, so callers can pose the same geometry
/// independently.
class wasm_mesh {
  std::shared_ptr<mesh_data> _data;

public:
  wasm_mesh() : _data(std::make_shared<mesh_data>()) {}
  wasm_mesh(const wasm_mesh &) = default;
  wasm_mesh(wasm_mesh &&) = default;
  auto operator=(const wasm_mesh &) -> wasm_mesh & = default;
  auto operator=(wasm_mesh &&) -> wasm_mesh & = default;

  // -- Factories --

  static auto create(wasm_ndarray<int> faces, wasm_ndarray<float> points)
      -> wasm_mesh {
    wasm_mesh m;
    m._data->set_faces(std::move(faces));
    m._data->set_points(std::move(points));
    return m;
  }

  static auto from_buffers(tf::buffer<int> &&faces, tf::buffer<float> &&points)
      -> wasm_mesh {
    wasm_mesh m;
    m._data->assign_faces(std::move(faces));
    m._data->assign_points(std::move(points));
    return m;
  }

  static auto from_polygons_buffer(tf::polygons_buffer<int, float, 3, 3> &&poly)
      -> wasm_mesh {
    return from_buffers(std::move(poly.faces_buffer().data_buffer()),
                        std::move(poly.points_buffer().data_buffer()));
  }

  // -- Data access --

  auto faces() const -> wasm_ndarray<int> { return _data->faces(); }
  auto points() const -> wasm_ndarray<float> { return _data->points(); }

  auto number_of_faces() const -> int { return _data->number_of_faces(); }
  auto number_of_points() const -> int { return _data->number_of_points(); }

  // -- Data mutation --

  auto set_faces(wasm_ndarray<int> faces) -> void {
    _data->set_faces(std::move(faces));
  }
  auto set_points(wasm_ndarray<float> points) -> void {
    _data->set_points(std::move(points));
  }

  // -- Transformation --

  auto has_transformation() const -> bool { return _data->has_transformation(); }
  auto transformation() const -> wasm_ndarray<float> {
    return _data->transformation();
  }
  auto transformation_view() const -> tf::transformation_view<float, 3> {
    return _data->transformation_view();
  }
  auto set_transformation(const wasm_ndarray<float> &t) -> void {
    _data->set_transformation(t);
  }
  auto clear_transformation() -> void { _data->clear_transformation(); }

  // -- Lazy-build accessors (non-const: trigger ensure internally) --

  auto tree() -> const tf::aabb_tree<int, float, 3> & { return _data->tree(); }
  auto half_edges() -> tf::half_edges<int> & { return _data->half_edges(); }

  // Void variant for JS / embind (the const-ref tree() can't cross to JS).
  auto build_tree() -> void { (void)_data->tree(); }

  auto face_membership() -> wasm_offset_blocked_buffer<int, int> {
    return _data->face_membership();
  }
  auto manifold_edge_link() -> wasm_ndarray<int> {
    return _data->manifold_edge_link();
  }
  auto face_link() -> wasm_offset_blocked_buffer<int, int> {
    return _data->face_link();
  }
  auto vertex_link() -> wasm_offset_blocked_buffer<int, int> {
    return _data->vertex_link();
  }
  auto normals() -> wasm_ndarray<float> { return _data->normals(); }
  auto point_normals() -> wasm_ndarray<float> { return _data->point_normals(); }

  // -- Setters (bypass lazy build) --

  auto set_face_membership(wasm_offset_blocked_buffer<int, int> fm) -> void {
    _data->set_face_membership(std::move(fm));
  }
  auto set_vertex_link(wasm_offset_blocked_buffer<int, int> vl) -> void {
    _data->set_vertex_link(std::move(vl));
  }
  auto set_face_link(wasm_offset_blocked_buffer<int, int> fl) -> void {
    _data->set_face_link(std::move(fl));
  }
  auto set_manifold_edge_link(wasm_ndarray<int> mel) -> void {
    _data->set_manifold_edge_link(std::move(mel));
  }
  auto set_normals(wasm_ndarray<float> n) -> void {
    _data->set_normals(std::move(n));
  }
  auto set_point_normals(wasm_ndarray<float> pn) -> void {
    _data->set_point_normals(std::move(pn));
  }
  auto set_half_edges(tf::half_edges<int> &&he) -> void {
    _data->set_half_edges(std::move(he));
  }

  // -- Range views (used by C++ pipelines) --

  auto faces_range() const { return _data->faces_range(); }
  auto points_range() const { return _data->points_range(); }
  auto polygons_range() const { return _data->polygons_range(); }
  auto face_membership_range() { return _data->face_membership_range(); }
  auto face_link_range() { return _data->face_link_range(); }
  auto vertex_link_range() { return _data->vertex_link_range(); }
  auto manifold_edge_link_range() { return _data->manifold_edge_link_range(); }

  template <typename Fn> auto with_form(Fn &&fn) -> decltype(auto) {
    return _data->with_form(std::forward<Fn>(fn));
  }

  // -- Shallow copy: forks data (new mesh_data, member-wise copy; storage
  //    still shared via wasm_ndarray's internal shared_ptr) and clears the
  //    transformation, so the copy can take a different pose.

  auto shallow_copy() const -> wasm_mesh {
    wasm_mesh v;
    *v._data = *_data;
    v._data->clear_transformation();
    return v;
  }

  // -- Lifecycle --

  auto destroy() -> void { _data.reset(); }

  auto is_valid() const -> bool { return _data && _data->is_valid(); }

  // -- Cache state inspectors (diagnostic — not exposed in TS wrapper) --

  auto is_tree_built() const -> bool { return _data->is_tree_built(); }
  auto is_tree_fresh() const -> bool { return _data->is_tree_fresh(); }
  auto is_half_edges_built() const -> bool { return _data->is_half_edges_built(); }
  auto is_half_edges_fresh() const -> bool { return _data->is_half_edges_fresh(); }
  auto is_face_membership_built() const -> bool { return _data->is_face_membership_built(); }
  auto is_face_membership_fresh() const -> bool { return _data->is_face_membership_fresh(); }
  auto is_manifold_edge_link_built() const -> bool { return _data->is_manifold_edge_link_built(); }
  auto is_manifold_edge_link_fresh() const -> bool { return _data->is_manifold_edge_link_fresh(); }
  auto is_face_link_built() const -> bool { return _data->is_face_link_built(); }
  auto is_face_link_fresh() const -> bool { return _data->is_face_link_fresh(); }
  auto is_vertex_link_built() const -> bool { return _data->is_vertex_link_built(); }
  auto is_vertex_link_fresh() const -> bool { return _data->is_vertex_link_fresh(); }
  auto is_normals_built() const -> bool { return _data->is_normals_built(); }
  auto is_normals_fresh() const -> bool { return _data->is_normals_fresh(); }
  auto is_point_normals_built() const -> bool { return _data->is_point_normals_built(); }
  auto is_point_normals_fresh() const -> bool { return _data->is_point_normals_fresh(); }
};

} // namespace ts
} // namespace tf
