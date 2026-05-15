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

#include "trueform/ts/core/point_cloud_data.hpp"
#include <memory>
#include <utility>

namespace tf {
namespace ts {

/// @brief WASM-resident point cloud handle.
///
/// Thin wrapper holding a std::shared_ptr<point_cloud_data<Real>>. All
/// methods forward to the inner data. Default copy = shared_ptr copy, so
/// handle copies share the same underlying data and any lazily built
/// caches.
///
/// shallow_copy() forks the data (new point_cloud_data, member-wise
/// copied — internal storage still shared via shared_ptr) and clears the
/// transformation, so callers can pose the same geometry independently.
template <typename Real>
class wasm_point_cloud {
  std::shared_ptr<point_cloud_data<Real>> _data;

public:
  wasm_point_cloud() : _data(std::make_shared<point_cloud_data<Real>>()) {}
  wasm_point_cloud(const wasm_point_cloud &) = default;
  wasm_point_cloud(wasm_point_cloud &&) = default;
  auto operator=(const wasm_point_cloud &) -> wasm_point_cloud & = default;
  auto operator=(wasm_point_cloud &&) -> wasm_point_cloud & = default;

  // -- Factory --

  static auto create(wasm_ndarray<Real> points) -> wasm_point_cloud {
    wasm_point_cloud pc;
    pc._data->set_points(std::move(points));
    return pc;
  }

  // -- Data access --

  auto points() const -> wasm_ndarray<Real> { return _data->points(); }

  auto number_of_points() const -> int { return _data->number_of_points(); }

  auto set_points(wasm_ndarray<Real> points) -> void {
    _data->set_points(std::move(points));
  }

  // -- Spatial tree (lazy build, cached) --

  auto tree() -> const tf::aabb_tree<int, Real, 3> & {
    return _data->tree();
  }

  // Void variant for JS / embind (the const-ref tree() can't cross to JS).
  auto build_tree() -> void { (void)_data->tree(); }

  // -- Vertex link (settable only) --

  auto vertex_link() const -> wasm_offset_blocked_buffer<int, int> {
    return _data->vertex_link();
  }

  auto has_vertex_link() const -> bool { return _data->has_vertex_link(); }

  auto set_vertex_link(wasm_offset_blocked_buffer<int, int> vl) -> void {
    _data->set_vertex_link(std::move(vl));
  }

  auto vertex_link_range() { return _data->vertex_link_range(); }

  // -- Normals (settable only) --

  auto normals() const -> wasm_ndarray<Real> { return _data->normals(); }

  auto has_normals() const -> bool { return _data->has_normals(); }

  auto set_normals(wasm_ndarray<Real> n) -> void {
    _data->set_normals(std::move(n));
  }

  // -- Transformation --

  auto has_transformation() const -> bool {
    return _data->has_transformation();
  }

  auto transformation() const -> wasm_ndarray<Real> {
    return _data->transformation();
  }

  auto transformation_view() const -> tf::transformation_view<Real, 3> {
    return _data->transformation_view();
  }

  auto set_transformation(const wasm_ndarray<Real> &t) -> void {
    _data->set_transformation(t);
  }

  auto clear_transformation() -> void { _data->clear_transformation(); }

  // -- Shallow copy: forks data (new point_cloud_data, member-wise copy;
  //    storage still shared via wasm_ndarray's internal shared_ptr) and
  //    clears the transformation, so the copy can take a different pose.

  auto shallow_copy() const -> wasm_point_cloud {
    wasm_point_cloud v;
    *v._data = *_data;
    v._data->clear_transformation();
    return v;
  }

  // -- Internal: trueform range view over raw data --

  auto points_range() const { return _data->points_range(); }

  // -- Spatial form construction --

  template <typename Fn> auto with_form(Fn &&fn) -> decltype(auto) {
    return _data->with_form(std::forward<Fn>(fn));
  }

  // -- Lifecycle --

  auto destroy() -> void { _data.reset(); }

  auto is_valid() const -> bool { return _data && _data->is_valid(); }

  // -- Cache state inspectors (diagnostic — not exposed in TS wrapper) --

  auto is_tree_built() const -> bool { return _data->is_tree_built(); }
  auto is_tree_fresh() const -> bool { return _data->is_tree_fresh(); }
};

} // namespace ts
} // namespace tf
