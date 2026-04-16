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

#include "trueform/core/points.hpp"
#include "trueform/core/transformation_view.hpp"
#include "trueform/core/policy/frame.hpp"
#include "trueform/spatial/aabb_tree.hpp"
#include "trueform/spatial/policy/tree.hpp"
#include "trueform/topology/vertex_link_like.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/core/wasm_offset_blocked_buffer.hpp"
#include <cstdint>
#include <memory>

namespace tf {
namespace ts {

/// @brief Inner data class for wasm_point_cloud (PIMPL).
///
/// Holds point coordinates and lazily-built spatial tree. Multiple
/// wasm_point_cloud handles can share the same point_cloud_data via
/// shared_ptr — that is how cache state is shared across copies (e.g.
/// async lambda captures).
///
/// ensure_tree() is private; access via the tree() accessor.
class point_cloud_data {
  wasm_ndarray<float> _points;         // [V, 3]
  wasm_ndarray<float> _transformation; // [4, 4] — empty when no transform

  // Spatial tree cache
  std::shared_ptr<tf::aabb_tree<int, float, 3>> _tree;
  uint32_t _tree_gen = 0;

  // Settable structures (no auto-compute)
  wasm_offset_blocked_buffer<int, int> _vl;
  wasm_ndarray<float> _normals; // [V, 3]

  // Generation counter
  uint32_t _points_gen = 1; // starts at 1 so tree is initially stale

public:
  point_cloud_data() = default;

  // -- Data access --

  auto points() const -> wasm_ndarray<float> { return _points; }

  auto number_of_points() const -> int {
    return _points.is_valid() ? _points.raw_shape()[0] : 0;
  }

  // -- Data mutation --

  auto set_points(wasm_ndarray<float> points) -> void {
    _points = std::move(points);
    ++_points_gen;
  }

  // -- Spatial tree (lazy build, cached) --

  auto tree() -> const tf::aabb_tree<int, float, 3> & {
    ensure_tree();
    return *_tree;
  }

  // -- Vertex link (settable only) --

  auto vertex_link() const -> wasm_offset_blocked_buffer<int, int> {
    return _vl;
  }

  auto has_vertex_link() const -> bool { return _vl.is_valid(); }

  auto set_vertex_link(wasm_offset_blocked_buffer<int, int> vl) -> void {
    _vl = std::move(vl);
  }

  auto vertex_link_range() {
    return tf::make_vertex_link_like(_vl.make_range());
  }

  // -- Normals (settable only) --

  auto normals() const -> wasm_ndarray<float> { return _normals; }

  auto has_normals() const -> bool { return _normals.is_valid(); }

  auto set_normals(wasm_ndarray<float> n) -> void { _normals = std::move(n); }

  // -- Transformation --

  auto has_transformation() const -> bool {
    return _transformation.is_valid();
  }

  auto transformation() const -> wasm_ndarray<float> {
    return _transformation;
  }

  auto transformation_view() const -> tf::transformation_view<float, 3> {
    return tf::make_transformation_view<3>(
        const_cast<float *>(_transformation.raw_data()));
  }

  auto set_transformation(const wasm_ndarray<float> &t) -> void {
    _transformation = t;
  }
  auto clear_transformation() -> void { _transformation.destroy(); }

  // -- Internal: trueform range view over raw data --

  auto points_range() const {
    return tf::make_points<3>(_points.make_range());
  }

  // -- Spatial form construction --

  template <typename Fn> auto with_form(Fn &&fn) -> decltype(auto) {
    ensure_tree();
    if (has_transformation()) {
      auto form =
          points_range() | tf::tag(*_tree) | tf::tag(transformation_view());
      return fn(form);
    } else {
      auto form = points_range() | tf::tag(*_tree);
      return fn(form);
    }
  }

  // -- Lifecycle --

  auto destroy() -> void {
    _points.destroy();
    _transformation.destroy();
    _vl.destroy();
    _normals.destroy();
    _tree.reset();
  }

  auto is_valid() const -> bool { return _points.is_valid(); }

  // -- Cache state inspectors (diagnostic) --

  auto is_tree_built() const -> bool { return _tree != nullptr; }
  auto is_tree_fresh() const -> bool {
    return _tree != nullptr && _tree_gen == _points_gen;
  }

private:
  auto ensure_tree() -> void {
    if (_tree && _tree_gen == _points_gen)
      return;
    _tree = std::make_shared<tf::aabb_tree<int, float, 3>>(
        points_range(), tf::config_tree(4, 4));
    _tree_gen = _points_gen;
  }
};

} // namespace ts
} // namespace tf
