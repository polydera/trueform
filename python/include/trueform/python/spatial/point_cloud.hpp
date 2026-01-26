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

#include "point_cloud_data.hpp"
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <trueform/core/transformation_view.hpp>

namespace tf::py {

template <typename RealT, std::size_t Dims>
class point_cloud_wrapper {
public:
  using data_type = point_cloud_data_wrapper<RealT, Dims>;

  point_cloud_wrapper() : _data{std::make_shared<data_type>()} {}

  point_cloud_wrapper(
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          points_array)
      : _data{std::make_shared<data_type>(points_array)} {}

  // Constructor from shared data
  point_cloud_wrapper(std::shared_ptr<data_type> data) : _data{std::move(data)} {}

  // Create a new wrapper sharing the same data (no transformation)
  auto shared_view() const -> point_cloud_wrapper {
    return point_cloud_wrapper{_data};
  }

  // Access to shared data
  auto data() -> std::shared_ptr<data_type> & { return _data; }
  auto data() const -> const std::shared_ptr<data_type> & { return _data; }

  // Forward primitive range creation to data
  auto make_primitive_range() { return _data->make_primitive_range(); }
  auto make_primitive_range() const { return _data->make_primitive_range(); }

  // Build methods (idempotent - only build if needed)
  auto build_tree() -> void { _data->build_tree(); }

  // Getter (auto-build if needed)
  auto tree() -> tf::aabb_tree<int, RealT, Dims> & { return _data->tree(); }

  // Has check
  auto has_tree() const -> bool { return _data->has_tree(); }

  // Data array accessors
  auto size() const -> std::size_t { return _data->size(); }
  auto dims() const -> std::size_t { return _data->dims(); }

  auto points_array() const
      -> nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>> {
    return _data->points_array();
  }
  auto set_points_array(
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          points_array) -> void {
    _data->set_points_array(points_array);
  }

  auto mark_modified() -> void { _data->mark_modified(); }

  // Transformation is local to this wrapper (not shared)
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
  std::shared_ptr<data_type> _data;
  std::optional<nanobind::ndarray<nanobind::numpy, RealT,
                                  nanobind::shape<Dims + 1, Dims + 1>>>
      _transformation;
};

} // namespace tf::py
