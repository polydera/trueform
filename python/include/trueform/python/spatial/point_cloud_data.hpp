/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/spatial/tree_config.hpp>

namespace tf::py {

template <typename RealT, std::size_t Dims>
class point_cloud_data_wrapper {
public:
  point_cloud_data_wrapper() = default;

  point_cloud_data_wrapper(
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          points_array)
      : _points_array{points_array} {}

  // Create view into Python-owned array
  auto make_primitive_range() {
    RealT *data = static_cast<RealT *>(_points_array.data());
    std::size_t count = _points_array.shape(0) * Dims;
    return tf::make_points<Dims>(tf::make_range(data, count));
  }

  auto make_primitive_range() const {
    const RealT *data = static_cast<const RealT *>(_points_array.data());
    std::size_t count = _points_array.shape(0) * Dims;
    return tf::make_points<Dims>(tf::make_range(data, count));
  }

  // Tree management
  auto rebuild_tree() -> void {
    if (!_tree) {
      _tree = std::make_unique<tf::aabb_tree<int, RealT, Dims>>();
    }
    auto pts = make_primitive_range();
    *_tree = tf::aabb_tree<int, RealT, Dims>(pts, tf::config_tree(4, 4));
    _tree_modified = false;
  }

  auto ensure_tree() -> void {
    if (!_tree || _tree_modified) {
      rebuild_tree();
      _tree_modified = false;
    }
  }

  auto mark_modified() -> void { _tree_modified = true; }

  auto clear_tree() -> void { _tree.reset(); }

  auto has_tree() const -> bool { return _tree != nullptr; }

  auto size() const -> std::size_t { return _points_array.shape(0); }

  auto dims() const -> std::size_t { return Dims; }

  auto tree() -> tf::aabb_tree<int, RealT, Dims> & {
    ensure_tree();
    return *_tree;
  }

  auto tree() const -> const tf::aabb_tree<int, RealT, Dims> & {
    if (!_tree)
      throw std::runtime_error("Tree not built");
    return *_tree;
  }

  auto points_array() const
      -> nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>> {
    return _points_array;
  }

  auto set_points_array(
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          points_array) -> void {
    _points_array = points_array;
    mark_modified();
  }

private:
  nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
      _points_array;
  std::unique_ptr<tf::aabb_tree<int, RealT, Dims>> _tree;
  bool _tree_modified = false;
};

} // namespace tf::py
