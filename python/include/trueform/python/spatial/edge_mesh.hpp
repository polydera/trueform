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

#include "edge_mesh_data.hpp"
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <trueform/core/transformation_view.hpp>

namespace tf::py {

template <typename Index, typename RealT, std::size_t Dims>
class edge_mesh_wrapper {
public:
  using data_type = edge_mesh_data_wrapper<Index, RealT, Dims>;

  edge_mesh_wrapper() : _data{std::make_shared<data_type>()} {}

  edge_mesh_wrapper(
      nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, 2>>
          edges_array,
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          points_array)
      : _data{std::make_shared<data_type>(edges_array, points_array)} {}

  // Constructor from shared data
  edge_mesh_wrapper(std::shared_ptr<data_type> data) : _data{std::move(data)} {}

  // Create a new wrapper sharing the same data (no transformation)
  auto shared_view() const -> edge_mesh_wrapper {
    return edge_mesh_wrapper{_data};
  }

  // Access to shared data
  auto data() -> std::shared_ptr<data_type> & { return _data; }
  auto data() const -> const std::shared_ptr<data_type> & { return _data; }

  // Forward primitive range creation to data
  auto make_primitive_range() { return _data->make_primitive_range(); }
  auto make_primitive_range() const { return _data->make_primitive_range(); }

  // Build methods (idempotent - only build if needed)
  auto build_tree() -> void { _data->build_tree(); }
  auto build_vertex_link() -> void { _data->build_vertex_link(); }
  auto build_edge_membership() -> void { _data->build_edge_membership(); }

  // Getters (auto-build if needed)
  auto tree() -> tf::aabb_tree<Index, RealT, Dims> & { return _data->tree(); }
  auto vertex_link() { return _data->vertex_link(); }
  auto edge_membership() { return _data->edge_membership(); }

  // Has checks
  auto has_tree() const -> bool { return _data->has_tree(); }
  auto has_vertex_link() const -> bool { return _data->has_vertex_link(); }
  auto has_edge_membership() const -> bool {
    return _data->has_edge_membership();
  }

  // Array accessors
  auto vertex_link_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    return _data->vertex_link_array();
  }

  auto edge_membership_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    return _data->edge_membership_array();
  }

  // Setters for pre-computed structures
  auto set_vertex_link(tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    _data->set_vertex_link(std::move(fm));
  }

  auto
  set_edge_membership(tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    _data->set_edge_membership(std::move(fm));
  }

  // Data array accessors
  auto number_of_edges() const -> std::size_t {
    return _data->number_of_edges();
  }
  auto number_of_points() const -> std::size_t {
    return _data->number_of_points();
  }
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

  auto edges_array() const
      -> nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, 2>> {
    return _data->edges_array();
  }
  auto set_edges_array(
      nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, 2>>
          edges_array) -> void {
    _data->set_edges_array(edges_array);
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
