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

  // Build methods (idempotent - only build if needed)
  auto build_tree() -> void {
    if (!_tree || _tree_modified) {
      do_build_tree();
    }
  }

  // Has check
  auto has_tree() const -> bool { return _tree != nullptr; }

  // Getter (auto-build if needed) - NO CONST VERSION
  auto tree() -> tf::aabb_tree<int, RealT, Dims> & {
    build_tree();
    return *_tree;
  }

  // Data array accessors
  auto size() const -> std::size_t { return _points_array.shape(0); }
  auto dims() const -> std::size_t { return Dims; }

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

  auto mark_modified() -> void { _tree_modified = true; }

private:
  auto do_build_tree() -> void {
    if (!_tree) {
      _tree = std::make_unique<tf::aabb_tree<int, RealT, Dims>>();
    }
    auto pts = make_primitive_range();
    *_tree = tf::aabb_tree<int, RealT, Dims>(pts, tf::config_tree(4, 4));
    _tree_modified = false;
  }

  nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
      _points_array;
  std::unique_ptr<tf::aabb_tree<int, RealT, Dims>> _tree;
  bool _tree_modified = false;
};

} // namespace tf::py
