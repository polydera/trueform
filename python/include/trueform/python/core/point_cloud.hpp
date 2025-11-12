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
#include <nanobind/stl/optional.h>
#include <optional>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/transformation_view.hpp>
#include <trueform/spatial/tree.hpp>
#include <trueform/spatial/tree_config.hpp>

namespace tf::py {

template <typename RealT, std::size_t Dims> class point_cloud_wrapper {
public:
  point_cloud_wrapper() = default;

  point_cloud_wrapper(
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
      _tree = std::make_unique<tf::tree<int, RealT, Dims>>();
    }
    auto pts = make_primitive_range();
    *_tree = tf::tree<int, RealT, Dims>(pts, tf::config_tree(4, 4));
  }

  auto ensure_tree() -> void {
    if (!_tree) {
      rebuild_tree();
    }
  }

  auto clear_tree() -> void { _tree.reset(); }

  auto has_tree() const -> bool { return _tree != nullptr; }

  auto size() const -> std::size_t { return _points_array.shape(0); }

  auto dims() const -> std::size_t { return Dims; }

  // Access to internal structures (opaque to Python)
  auto tree() -> tf::tree<int, RealT, Dims> & {
    ensure_tree();
    return *_tree;
  }

  auto tree() const -> const tf::tree<int, RealT, Dims> & {
    if (!_tree)
      throw std::runtime_error("Tree not built");
    return *_tree;
  }

  auto points_array() const
      -> nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>> {
    return _points_array;
  }

  auto has_transformation() const -> bool {
    return _transformation.has_value();
  }

  auto transformation() const
      -> std::optional<nanobind::ndarray<nanobind::numpy, RealT,
                                         nanobind::shape<Dims + 1, Dims + 1>>> {
    return _transformation;
  }

  auto transformation_view() const {
    const auto &trans = *_transformation;
    return tf::make_transformation_view<Dims>(trans.data());
  }

  auto set_transformation(nanobind::ndarray<nanobind::numpy, RealT,
                                            nanobind::shape<Dims + 1, Dims + 1>>
                              transformation_array) -> void {
    _transformation = transformation_array;
  }

  auto clear_transformation() -> void { _transformation.reset(); }

private:
  nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
      _points_array;
  std::optional<nanobind::ndarray<nanobind::numpy, RealT,
                                  nanobind::shape<Dims + 1, Dims + 1>>>
      _transformation;
  std::unique_ptr<tf::tree<int, RealT, Dims>> _tree;
};

// Forward declaration of registration function
auto register_point_cloud(nanobind::module_ &m) -> void;

} // namespace tf::py
