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

#include "mesh_data.hpp"
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <trueform/core/transformation_view.hpp>

namespace tf::py {

template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
class mesh_wrapper {
public:
  using data_type = mesh_data_wrapper<Index, RealT, Ngon, Dims>;

  mesh_wrapper() : _data{std::make_shared<data_type>()} {}

  mesh_wrapper(
      nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>>
          faces_array,
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          points_array)
      : _data{std::make_shared<data_type>(faces_array, points_array)} {}

  // Constructor from shared data
  mesh_wrapper(std::shared_ptr<data_type> data) : _data{std::move(data)} {}

  // Create a new wrapper sharing the same data (no transformation)
  auto shared_view() const -> mesh_wrapper { return mesh_wrapper{_data}; }

  // Access to shared data
  auto data() -> std::shared_ptr<data_type> & { return _data; }
  auto data() const -> const std::shared_ptr<data_type> & { return _data; }

  // Forward primitive range creation to data
  auto make_primitive_range() { return _data->make_primitive_range(); }
  auto make_primitive_range() const { return _data->make_primitive_range(); }

  // Build methods (idempotent - only build if needed)
  auto build_tree() -> void { _data->build_tree(); }
  auto build_face_membership() -> void { _data->build_face_membership(); }
  auto build_face_link() -> void { _data->build_face_link(); }
  auto build_vertex_link() -> void { _data->build_vertex_link(); }
  auto build_manifold_edge_link() -> void { _data->build_manifold_edge_link(); }
  auto build_normals() -> void { _data->build_normals(); }
  auto build_point_normals() -> void { _data->build_point_normals(); }

  // Getters (auto-build if needed)
  auto tree() -> tf::aabb_tree<Index, RealT, Dims> & { return _data->tree(); }
  auto face_membership() { return _data->face_membership(); }
  auto face_link() { return _data->face_link(); }
  auto vertex_link() { return _data->vertex_link(); }
  auto manifold_edge_link() { return _data->manifold_edge_link(); }
  auto normals() { return _data->normals(); }
  auto point_normals() { return _data->point_normals(); }

  // Has checks
  auto has_tree() const -> bool { return _data->has_tree(); }
  auto has_face_membership() const -> bool {
    return _data->has_face_membership();
  }
  auto has_face_link() const -> bool { return _data->has_face_link(); }
  auto has_vertex_link() const -> bool { return _data->has_vertex_link(); }
  auto has_manifold_edge_link() const -> bool {
    return _data->has_manifold_edge_link();
  }
  auto has_normals() const -> bool { return _data->has_normals(); }
  auto has_point_normals() const -> bool { return _data->has_point_normals(); }

  // Array accessors
  auto face_membership_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    return _data->face_membership_array();
  }

  auto face_link_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    return _data->face_link_array();
  }

  auto vertex_link_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    return _data->vertex_link_array();
  }

  auto manifold_edge_link_array() -> const
      nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>> & {
    return _data->manifold_edge_link_array();
  }

  auto normals_array()
      -> const nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>> & {
    return _data->normals_array();
  }

  auto point_normals_array()
      -> const nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>> & {
    return _data->point_normals_array();
  }

  // Setters for pre-computed structures
  auto set_face_membership(
      tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    _data->set_face_membership(std::move(fm));
  }

  auto set_face_link(tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    _data->set_face_link(std::move(fm));
  }

  auto set_vertex_link(tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    _data->set_vertex_link(std::move(fm));
  }

  auto set_manifold_edge_link(
      nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>>
          mel) {
    _data->set_manifold_edge_link(mel);
  }

  auto set_normals(
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          normals) {
    _data->set_normals(normals);
  }

  auto set_point_normals(
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          point_normals) {
    _data->set_point_normals(point_normals);
  }

  // Data array accessors
  auto number_of_faces() const -> std::size_t {
    return _data->number_of_faces();
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

  auto faces_array() const
      -> nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>> {
    return _data->faces_array();
  }
  auto set_faces_array(
      nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>>
          faces_array) -> void {
    _data->set_faces_array(faces_array);
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

// Specialization for dynamic-size polygons (n-gons)
template <typename Index, typename RealT, std::size_t Dims>
class mesh_wrapper<Index, RealT, tf::dynamic_size, Dims> {
public:
  using data_type = mesh_data_wrapper<Index, RealT, tf::dynamic_size, Dims>;

  mesh_wrapper() : _data{std::make_shared<data_type>()} {}

  mesh_wrapper(
      tf::py::offset_blocked_array_wrapper<Index, Index> faces_array,
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          points_array)
      : _data{std::make_shared<data_type>(std::move(faces_array),
                                          points_array)} {}

  // Constructor from shared data
  mesh_wrapper(std::shared_ptr<data_type> data) : _data{std::move(data)} {}

  // Create a new wrapper sharing the same data (no transformation)
  auto shared_view() const -> mesh_wrapper { return mesh_wrapper{_data}; }

  // Access to shared data
  auto data() -> std::shared_ptr<data_type> & { return _data; }
  auto data() const -> const std::shared_ptr<data_type> & { return _data; }

  // Forward primitive range creation to data
  auto make_primitive_range() { return _data->make_primitive_range(); }
  auto make_primitive_range() const { return _data->make_primitive_range(); }

  // Build methods (idempotent - only build if needed)
  auto build_tree() -> void { _data->build_tree(); }
  auto build_face_membership() -> void { _data->build_face_membership(); }
  auto build_face_link() -> void { _data->build_face_link(); }
  auto build_vertex_link() -> void { _data->build_vertex_link(); }
  auto build_manifold_edge_link() -> void { _data->build_manifold_edge_link(); }
  auto build_normals() -> void { _data->build_normals(); }
  auto build_point_normals() -> void { _data->build_point_normals(); }

  // Getters (auto-build if needed)
  auto tree() -> tf::aabb_tree<Index, RealT, Dims> & { return _data->tree(); }
  auto face_membership() { return _data->face_membership(); }
  auto face_link() { return _data->face_link(); }
  auto vertex_link() { return _data->vertex_link(); }
  auto manifold_edge_link() { return _data->manifold_edge_link(); }
  auto normals() { return _data->normals(); }
  auto point_normals() { return _data->point_normals(); }

  // Has checks
  auto has_tree() const -> bool { return _data->has_tree(); }
  auto has_face_membership() const -> bool {
    return _data->has_face_membership();
  }
  auto has_face_link() const -> bool { return _data->has_face_link(); }
  auto has_vertex_link() const -> bool { return _data->has_vertex_link(); }
  auto has_manifold_edge_link() const -> bool {
    return _data->has_manifold_edge_link();
  }
  auto has_normals() const -> bool { return _data->has_normals(); }
  auto has_point_normals() const -> bool { return _data->has_point_normals(); }

  // Array accessors (all use offset_blocked_array_wrapper for dynamic size)
  auto face_membership_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    return _data->face_membership_array();
  }

  auto face_link_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    return _data->face_link_array();
  }

  auto vertex_link_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    return _data->vertex_link_array();
  }

  auto manifold_edge_link_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    return _data->manifold_edge_link_array();
  }

  auto normals_array()
      -> const nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>> & {
    return _data->normals_array();
  }

  auto point_normals_array()
      -> const nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>> & {
    return _data->point_normals_array();
  }

  // Setters for pre-computed structures
  auto set_face_membership(
      tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    _data->set_face_membership(std::move(fm));
  }

  auto set_face_link(tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    _data->set_face_link(std::move(fm));
  }

  auto set_vertex_link(tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    _data->set_vertex_link(std::move(fm));
  }

  auto set_manifold_edge_link(
      tf::py::offset_blocked_array_wrapper<Index, Index> mel) {
    _data->set_manifold_edge_link(std::move(mel));
  }

  auto set_normals(
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          normals) {
    _data->set_normals(normals);
  }

  auto set_point_normals(
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          point_normals) {
    _data->set_point_normals(point_normals);
  }

  // Data array accessors
  auto number_of_faces() const -> std::size_t {
    return _data->number_of_faces();
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

  // For dynamic size, faces are stored as offset_blocked_array
  auto faces_array() const
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    return _data->faces_array();
  }
  auto set_faces_array(
      tf::py::offset_blocked_array_wrapper<Index, Index> faces_array) -> void {
    _data->set_faces_array(std::move(faces_array));
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
