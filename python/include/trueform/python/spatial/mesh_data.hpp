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

#include "../core/offset_blocked_array.hpp"
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/offset_block_buffer.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/polygons.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/geometry/compute_normals.hpp>
#include <trueform/geometry/compute_point_normals.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/spatial/tree_config.hpp>
#include <trueform/topology/face_link.hpp>
#include <trueform/topology/face_membership.hpp>
#include <trueform/topology/manifold_edge_link.hpp>
#include <trueform/topology/vertex_link.hpp>

namespace tf::py {

template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
class mesh_data_wrapper {
public:
  mesh_data_wrapper() = default;

  mesh_data_wrapper(
      nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>>
          faces_array,
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          points_array)
      : _faces_array{faces_array}, _points_array{points_array} {}

  // Create view into Python-owned array
  auto make_primitive_range() {
    RealT *data_pts = static_cast<RealT *>(_points_array.data());
    std::size_t count_pts = _points_array.shape(0) * Dims;
    auto pts = tf::make_points<Dims>(tf::make_range(data_pts, count_pts));
    Index *data_fcs = static_cast<Index *>(_faces_array.data());
    std::size_t count_fcs = _faces_array.shape(0) * Ngon;
    auto faces =
        tf::make_blocked_range<Ngon>(tf::make_range(data_fcs, count_fcs));
    return tf::make_polygons(faces, pts);
  }

  auto make_primitive_range() const {
    const RealT *data_pts = static_cast<const RealT *>(_points_array.data());
    std::size_t count_pts = _points_array.shape(0) * Dims;
    auto pts = tf::make_points<Dims>(tf::make_range(data_pts, count_pts));
    const Index *data_fcs = static_cast<const Index *>(_faces_array.data());
    std::size_t count_fcs = _faces_array.shape(0) * Ngon;
    auto faces =
        tf::make_blocked_range<Ngon>(tf::make_range(data_fcs, count_fcs));
    return tf::make_polygons(faces, pts);
  }

  // Build methods (idempotent - only build if needed)
  auto build_tree() -> void {
    if (!_tree || _tree_modified) {
      do_build_tree();
    }
  }

  auto build_face_membership() -> void {
    if (!_face_membership_array || _face_membership_modified) {
      do_build_face_membership();
    }
  }

  auto build_face_link() -> void {
    build_face_membership();
    if (!_face_link_array || _face_link_modified) {
      do_build_face_link();
    }
  }

  auto build_vertex_link() -> void {
    build_face_membership();
    if (!_vertex_link_array || _vertex_link_modified) {
      do_build_vertex_link();
    }
  }

  auto build_manifold_edge_link() -> void {
    build_face_membership();
    if (!_manifold_edge_link_array || _manifold_edge_link_modified) {
      do_build_manifold_edge_link();
    }
  }

  auto build_normals() -> void {
    if (!_normals_array || _normals_modified) {
      do_build_normals();
    }
  }

  auto build_point_normals() -> void {
    build_face_membership();
    build_normals();
    if (!_point_normals_array || _point_normals_modified) {
      do_build_point_normals();
    }
  }

  // Getters (auto-build if needed)
  auto tree() -> tf::aabb_tree<Index, RealT, Dims> & {
    build_tree();
    return *_tree;
  }

  auto face_membership() {
    build_face_membership();
    return tf::make_face_membership_like(_face_membership_array->make_range());
  }

  auto face_link() {
    build_face_link();
    return tf::make_face_link_like(_face_link_array->make_range());
  }

  auto vertex_link() {
    build_vertex_link();
    return tf::make_vertex_link_like(_vertex_link_array->make_range());
  }

  auto manifold_edge_link() {
    build_manifold_edge_link();
    return make_manifold_edge_link_view();
  }

  auto normals() {
    build_normals();
    return make_normals_view();
  }

  auto point_normals() {
    build_point_normals();
    return make_point_normals_view();
  }

  // Has checks
  auto has_tree() const -> bool { return _tree != nullptr && !_tree_modified; }

  auto has_face_membership() const -> bool {
    return _face_membership_array != nullptr && !_face_membership_modified;
  }

  auto has_face_link() const -> bool {
    return _face_link_array != nullptr && !_face_link_modified;
  }

  auto has_vertex_link() const -> bool {
    return _vertex_link_array != nullptr && !_vertex_link_modified;
  }

  auto has_manifold_edge_link() const -> bool {
    return _manifold_edge_link_array != nullptr &&
           !_manifold_edge_link_modified;
  }

  auto has_normals() const -> bool {
    return _normals_array != nullptr && !_normals_modified;
  }

  auto has_point_normals() const -> bool {
    return _point_normals_array != nullptr && !_point_normals_modified;
  }

  // Array accessors
  auto face_membership_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    build_face_membership();
    return *_face_membership_array;
  }

  auto face_link_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    build_face_link();
    return *_face_link_array;
  }

  auto vertex_link_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    build_vertex_link();
    return *_vertex_link_array;
  }

  auto manifold_edge_link_array() -> const
      nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>> & {
    build_manifold_edge_link();
    return *_manifold_edge_link_array;
  }

  auto normals_array() -> const
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>> & {
    build_normals();
    return *_normals_array;
  }

  auto point_normals_array() -> const
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>> & {
    build_point_normals();
    return *_point_normals_array;
  }

  // Setters for pre-computed structures
  auto
  set_face_membership(tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    if (!_face_membership_array)
      _face_membership_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              fm.offsets_array(), fm.data_array());
    else
      _face_membership_array->set_arrays(fm.offsets_array(), fm.data_array());
    _face_membership_modified = false;
  }

  auto set_face_link(tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    if (!_face_link_array)
      _face_link_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              fm.offsets_array(), fm.data_array());
    else
      _face_link_array->set_arrays(fm.offsets_array(), fm.data_array());
    _face_link_modified = false;
  }

  auto set_vertex_link(tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    if (!_vertex_link_array)
      _vertex_link_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              fm.offsets_array(), fm.data_array());
    else
      _vertex_link_array->set_arrays(fm.offsets_array(), fm.data_array());
    _vertex_link_modified = false;
  }

  auto set_manifold_edge_link(
      nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>>
          mel) {
    if (!_manifold_edge_link_array)
      _manifold_edge_link_array =
          std::make_unique<nanobind::ndarray<nanobind::numpy, Index,
                                             nanobind::shape<-1, Ngon>>>();
    *_manifold_edge_link_array = mel;
    _manifold_edge_link_modified = false;
  }

  auto set_normals(
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          normals) {
    if (!_normals_array)
      _normals_array =
          std::make_unique<nanobind::ndarray<nanobind::numpy, RealT,
                                             nanobind::shape<-1, Dims>>>();
    *_normals_array = normals;
    _normals_modified = false;
  }

  auto set_point_normals(
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          point_normals) {
    if (!_point_normals_array)
      _point_normals_array =
          std::make_unique<nanobind::ndarray<nanobind::numpy, RealT,
                                             nanobind::shape<-1, Dims>>>();
    *_point_normals_array = point_normals;
    _point_normals_modified = false;
  }

  // Data array accessors
  auto number_of_faces() const -> std::size_t { return _faces_array.shape(0); }

  auto number_of_points() const -> std::size_t {
    return _points_array.shape(0);
  }

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

  auto faces_array() const
      -> nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>> {
    return _faces_array;
  }

  auto set_faces_array(
      nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>>
          faces_array) -> void {
    _faces_array = faces_array;
    mark_modified();
  }

  auto mark_modified() -> void {
    _tree_modified = true;
    _face_membership_modified = true;
    _manifold_edge_link_modified = true;
    _face_link_modified = true;
    _vertex_link_modified = true;
    _normals_modified = true;
    _point_normals_modified = true;
  }

private:
  auto do_build_tree() -> void {
    if (!_tree) {
      _tree = std::make_unique<tf::aabb_tree<Index, RealT, Dims>>();
    }
    auto polys = make_primitive_range();
    *_tree = tf::aabb_tree<Index, RealT, Dims>(polys, tf::config_tree(4, 4));
    _tree_modified = false;
  }

  auto do_build_face_membership() -> void {
    auto polygons = make_primitive_range();
    tf::face_membership<Index> fm;
    fm.build(polygons);

    auto [offsets, data] = make_numpy_array(std::move(fm));

    if (!_face_membership_array) {
      _face_membership_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              offsets, data);
    } else {
      _face_membership_array->set_arrays(offsets, data);
    }
    _face_membership_modified = false;
  }

  auto do_build_face_link() -> void {
    auto polygons = make_primitive_range();
    tf::face_link<Index> fl;
    fl.build(polygons.faces(), face_membership());

    auto [offsets, data] = make_numpy_array(std::move(fl));

    if (!_face_link_array) {
      _face_link_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              offsets, data);
    } else {
      _face_link_array->set_arrays(offsets, data);
    }
    _face_link_modified = false;
  }

  auto do_build_vertex_link() -> void {
    auto polygons = make_primitive_range();
    tf::vertex_link<Index> vl;
    vl.build(polygons.faces(), face_membership());

    auto [offsets, data] = make_numpy_array(std::move(vl));

    if (!_vertex_link_array) {
      _vertex_link_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              offsets, data);
    } else {
      _vertex_link_array->set_arrays(offsets, data);
    }
    _vertex_link_modified = false;
  }

  auto do_build_manifold_edge_link() -> void {
    if (!_manifold_edge_link_array)
      _manifold_edge_link_array =
          std::make_unique<nanobind::ndarray<nanobind::numpy, Index,
                                             nanobind::shape<-1, Ngon>>>();

    const Index *data_fcs = static_cast<const Index *>(_faces_array.data());
    std::size_t count_fcs = _faces_array.shape(0) * Ngon;
    auto faces =
        tf::make_blocked_range<Ngon>(tf::make_range(data_fcs, count_fcs));
    tf::blocked_buffer<Index, Ngon> buff;
    buff.allocate(faces.size());
    tf::topology::compute_manifold_edge_link<Index>(faces, face_membership(),
                                                    buff);
    *_manifold_edge_link_array = make_numpy_array(std::move(buff));
    _manifold_edge_link_modified = false;
  }

  auto make_manifold_edge_link_view() {
    const Index *data_mel =
        static_cast<const Index *>(_manifold_edge_link_array->data());
    std::size_t count_mel = _manifold_edge_link_array->shape(0) * Ngon;
    struct dref_t {
      auto operator()(Index i) const -> tf::manifold_edge_peer<Index> {
        return {i};
      }
    };
    auto r =
        tf::make_mapped_range(tf::make_range(data_mel, count_mel), dref_t{});
    auto mel = tf::make_blocked_range<Ngon>(r);
    return tf::make_manifold_edge_link_like(mel);
  }

  auto do_build_normals() -> void {
      if (!_normals_array)
        _normals_array =
            std::make_unique<nanobind::ndarray<nanobind::numpy, RealT,
                                               nanobind::shape<-1, Dims>>>();
    if constexpr (Dims == 3) {
      auto polygons = make_primitive_range();
      auto normals_buff = tf::compute_normals(polygons);
      *_normals_array = make_numpy_array<nanobind::shape<-1, Dims>>(
          std::move(normals_buff.data_buffer()), {normals_buff.size(), Dims});
    }
      _normals_modified = false;
  }

  auto do_build_point_normals() -> void {
    if (!_point_normals_array)
      _point_normals_array =
          std::make_unique<nanobind::ndarray<nanobind::numpy, RealT,
                                             nanobind::shape<-1, Dims>>>();
    if constexpr (Dims == 3) {
    auto polygons = make_primitive_range();
    // Tag polygons with face_membership and normals for compute_point_normals
    auto tagged = polygons | tf::tag(face_membership()) |
                  tf::tag_normals(make_normals_view());
    auto point_normals_buff = tf::compute_point_normals(tagged);
    *_point_normals_array = make_numpy_array<nanobind::shape<-1, Dims>>(
        std::move(point_normals_buff.data_buffer()),
        {point_normals_buff.size(), Dims});
    }
    _point_normals_modified = false;

  }

  auto make_normals_view() {
    const RealT *data = static_cast<const RealT *>(_normals_array->data());
    std::size_t count = _normals_array->shape(0) * Dims;
    return tf::make_unit_vectors<Dims>(tf::make_range(data, count));
  }

  auto make_point_normals_view() {
    const RealT *data =
        static_cast<const RealT *>(_point_normals_array->data());
    std::size_t count = _point_normals_array->shape(0) * Dims;
    return tf::make_unit_vectors<Dims>(tf::make_range(data, count));
  }

  nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>>
      _faces_array;
  nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
      _points_array;
  std::unique_ptr<tf::aabb_tree<Index, RealT, Dims>> _tree;
  std::unique_ptr<tf::py::offset_blocked_array_wrapper<Index, Index>>
      _face_membership_array;
  std::unique_ptr<
      nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>>>
      _manifold_edge_link_array;
  std::unique_ptr<tf::py::offset_blocked_array_wrapper<Index, Index>>
      _face_link_array;
  std::unique_ptr<tf::py::offset_blocked_array_wrapper<Index, Index>>
      _vertex_link_array;
  std::unique_ptr<
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>>
      _normals_array;
  std::unique_ptr<
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>>
      _point_normals_array;
  bool _tree_modified = false;
  bool _face_membership_modified = false;
  bool _manifold_edge_link_modified = false;
  bool _face_link_modified = false;
  bool _vertex_link_modified = false;
  bool _normals_modified = false;
  bool _point_normals_modified = false;
};

// Specialization for dynamic-size polygons (n-gons)
template <typename Index, typename RealT, std::size_t Dims>
class mesh_data_wrapper<Index, RealT, tf::dynamic_size, Dims> {
public:
  mesh_data_wrapper() = default;

  mesh_data_wrapper(
      tf::py::offset_blocked_array_wrapper<Index, Index> faces_array,
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          points_array)
      : _faces_array{std::make_unique<
            tf::py::offset_blocked_array_wrapper<Index, Index>>(
            faces_array.offsets_array(), faces_array.data_array())},
        _points_array{points_array} {}

  // Create view into Python-owned array
  auto make_primitive_range() {
    RealT *data_pts = static_cast<RealT *>(_points_array.data());
    std::size_t count_pts = _points_array.shape(0) * Dims;
    auto pts = tf::make_points<Dims>(tf::make_range(data_pts, count_pts));
    auto faces = _faces_array->make_range();
    return tf::make_polygons(faces, pts);
  }

  auto make_primitive_range() const {
    const RealT *data_pts = static_cast<const RealT *>(_points_array.data());
    std::size_t count_pts = _points_array.shape(0) * Dims;
    auto pts = tf::make_points<Dims>(tf::make_range(data_pts, count_pts));
    auto faces = _faces_array->make_range();
    return tf::make_polygons(faces, pts);
  }

  // Build methods (idempotent - only build if needed)
  auto build_tree() -> void {
    if (!_tree || _tree_modified) {
      do_build_tree();
    }
  }

  auto build_face_membership() -> void {
    if (!_face_membership_array || _face_membership_modified) {
      do_build_face_membership();
    }
  }

  auto build_face_link() -> void {
    build_face_membership();
    if (!_face_link_array || _face_link_modified) {
      do_build_face_link();
    }
  }

  auto build_vertex_link() -> void {
    build_face_membership();
    if (!_vertex_link_array || _vertex_link_modified) {
      do_build_vertex_link();
    }
  }

  auto build_manifold_edge_link() -> void {
    build_face_membership();
    if (!_manifold_edge_link_array || _manifold_edge_link_modified) {
      do_build_manifold_edge_link();
    }
  }

  auto build_normals() -> void {
    if (!_normals_array || _normals_modified) {
      do_build_normals();
    }
  }

  auto build_point_normals() -> void {
    build_face_membership();
    build_normals();
    if (!_point_normals_array || _point_normals_modified) {
      do_build_point_normals();
    }
  }

  // Getters (auto-build if needed)
  auto tree() -> tf::aabb_tree<Index, RealT, Dims> & {
    build_tree();
    return *_tree;
  }

  auto face_membership() {
    build_face_membership();
    return tf::make_face_membership_like(_face_membership_array->make_range());
  }

  auto face_link() {
    build_face_link();
    return tf::make_face_link_like(_face_link_array->make_range());
  }

  auto vertex_link() {
    build_vertex_link();
    return tf::make_vertex_link_like(_vertex_link_array->make_range());
  }

  auto manifold_edge_link() {
    build_manifold_edge_link();
    return make_manifold_edge_link_view();
  }

  auto normals() {
    build_normals();
    return make_normals_view();
  }

  auto point_normals() {
    build_point_normals();
    return make_point_normals_view();
  }

  // Has checks
  auto has_tree() const -> bool { return _tree != nullptr && !_tree_modified; }

  auto has_face_membership() const -> bool {
    return _face_membership_array != nullptr && !_face_membership_modified;
  }

  auto has_face_link() const -> bool {
    return _face_link_array != nullptr && !_face_link_modified;
  }

  auto has_vertex_link() const -> bool {
    return _vertex_link_array != nullptr && !_vertex_link_modified;
  }

  auto has_manifold_edge_link() const -> bool {
    return _manifold_edge_link_array != nullptr &&
           !_manifold_edge_link_modified;
  }

  auto has_normals() const -> bool {
    return _normals_array != nullptr && !_normals_modified;
  }

  auto has_point_normals() const -> bool {
    return _point_normals_array != nullptr && !_point_normals_modified;
  }

  // Array accessors
  auto face_membership_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    build_face_membership();
    return *_face_membership_array;
  }

  auto face_link_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    build_face_link();
    return *_face_link_array;
  }

  auto vertex_link_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    build_vertex_link();
    return *_vertex_link_array;
  }

  auto manifold_edge_link_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    build_manifold_edge_link();
    return *_manifold_edge_link_array;
  }

  auto normals_array() -> const
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>> & {
    build_normals();
    return *_normals_array;
  }

  auto point_normals_array() -> const
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>> & {
    build_point_normals();
    return *_point_normals_array;
  }

  // Setters for pre-computed structures
  auto
  set_face_membership(tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    if (!_face_membership_array)
      _face_membership_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              fm.offsets_array(), fm.data_array());
    else
      _face_membership_array->set_arrays(fm.offsets_array(), fm.data_array());
    _face_membership_modified = false;
  }

  auto set_face_link(tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    if (!_face_link_array)
      _face_link_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              fm.offsets_array(), fm.data_array());
    else
      _face_link_array->set_arrays(fm.offsets_array(), fm.data_array());
    _face_link_modified = false;
  }

  auto set_vertex_link(tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    if (!_vertex_link_array)
      _vertex_link_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              fm.offsets_array(), fm.data_array());
    else
      _vertex_link_array->set_arrays(fm.offsets_array(), fm.data_array());
    _vertex_link_modified = false;
  }

  auto set_manifold_edge_link(
      tf::py::offset_blocked_array_wrapper<Index, Index> mel) {
    if (!_manifold_edge_link_array)
      _manifold_edge_link_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              mel.offsets_array(), mel.data_array());
    else
      _manifold_edge_link_array->set_arrays(mel.offsets_array(),
                                            mel.data_array());
    _manifold_edge_link_modified = false;
  }

  auto set_normals(
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          normals) {
    if (!_normals_array)
      _normals_array =
          std::make_unique<nanobind::ndarray<nanobind::numpy, RealT,
                                             nanobind::shape<-1, Dims>>>();
    *_normals_array = normals;
    _normals_modified = false;
  }

  auto set_point_normals(
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          point_normals) {
    if (!_point_normals_array)
      _point_normals_array =
          std::make_unique<nanobind::ndarray<nanobind::numpy, RealT,
                                             nanobind::shape<-1, Dims>>>();
    *_point_normals_array = point_normals;
    _point_normals_modified = false;
  }

  // Data array accessors
  auto number_of_faces() const -> std::size_t { return _faces_array->size(); }

  auto number_of_points() const -> std::size_t {
    return _points_array.shape(0);
  }

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

  auto faces_array() const
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    return *_faces_array;
  }

  auto set_faces_array(
      tf::py::offset_blocked_array_wrapper<Index, Index> faces_array) -> void {
    if (!_faces_array)
      _faces_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              faces_array.offsets_array(), faces_array.data_array());
    else
      _faces_array->set_arrays(faces_array.offsets_array(),
                               faces_array.data_array());
    mark_modified();
  }

  auto mark_modified() -> void {
    _tree_modified = true;
    _face_membership_modified = true;
    _manifold_edge_link_modified = true;
    _face_link_modified = true;
    _vertex_link_modified = true;
    _normals_modified = true;
    _point_normals_modified = true;
  }

private:
  auto do_build_tree() -> void {
    if (!_tree) {
      _tree = std::make_unique<tf::aabb_tree<Index, RealT, Dims>>();
    }
    auto polys = make_primitive_range();
    *_tree = tf::aabb_tree<Index, RealT, Dims>(polys, tf::config_tree(4, 4));
    _tree_modified = false;
  }

  auto do_build_face_membership() -> void {
    auto polygons = make_primitive_range();
    tf::face_membership<Index> fm;
    fm.build(polygons);

    auto [offsets, data] = make_numpy_array(std::move(fm));

    if (!_face_membership_array) {
      _face_membership_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              offsets, data);
    } else {
      _face_membership_array->set_arrays(offsets, data);
    }
    _face_membership_modified = false;
  }

  auto do_build_face_link() -> void {
    auto polygons = make_primitive_range();
    tf::face_link<Index> fl;
    fl.build(polygons.faces(), face_membership());

    auto [offsets, data] = make_numpy_array(std::move(fl));

    if (!_face_link_array) {
      _face_link_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              offsets, data);
    } else {
      _face_link_array->set_arrays(offsets, data);
    }
    _face_link_modified = false;
  }

  auto do_build_vertex_link() -> void {
    auto polygons = make_primitive_range();
    tf::vertex_link<Index> vl;
    vl.build(polygons.faces(), face_membership());

    auto [offsets, data] = make_numpy_array(std::move(vl));

    if (!_vertex_link_array) {
      _vertex_link_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              offsets, data);
    } else {
      _vertex_link_array->set_arrays(offsets, data);
    }
    _vertex_link_modified = false;
  }

  auto do_build_manifold_edge_link() -> void {
    auto faces = _faces_array->make_range();

    // Allocate buffer with same structure as faces (offsets are identical)
    tf::offset_block_buffer<Index, Index> buff;
    buff.offsets_buffer().allocate(_faces_array->offsets_array().size());
    buff.data_buffer().allocate(_faces_array->data_array().size());

    // Copy offsets from faces
    const Index *faces_offsets =
        static_cast<const Index *>(_faces_array->offsets_array().data());
    tf::parallel_copy(
        tf::make_range(faces_offsets, _faces_array->offsets_array().size()),
        buff.offsets_buffer());

    // Compute manifold edge link into the buffer
    tf::topology::compute_manifold_edge_link<Index>(faces, face_membership(),
                                                    buff);

    auto [offsets, data] = make_numpy_array(std::move(buff));

    if (!_manifold_edge_link_array) {
      _manifold_edge_link_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              offsets, data);
    } else {
      _manifold_edge_link_array->set_arrays(offsets, data);
    }
    _manifold_edge_link_modified = false;
  }

  auto do_build_normals() -> void {
    if (!_normals_array)
      _normals_array =
          std::make_unique<nanobind::ndarray<nanobind::numpy, RealT,
                                             nanobind::shape<-1, Dims>>>();
    if constexpr (Dims == 3) {
    auto polygons = make_primitive_range();
    auto normals_buff = tf::compute_normals(polygons);
    *_normals_array = make_numpy_array<nanobind::shape<-1, Dims>>(
        std::move(normals_buff.data_buffer()), {normals_buff.size(), Dims});
    }
    _normals_modified = false;
  }

  auto do_build_point_normals() -> void {
    if (!_point_normals_array)
      _point_normals_array =
          std::make_unique<nanobind::ndarray<nanobind::numpy, RealT,
                                             nanobind::shape<-1, Dims>>>();
    if constexpr (Dims == 3) {
    auto polygons = make_primitive_range();
    // Tag polygons with face_membership and normals for compute_point_normals
    auto tagged = polygons | tf::tag(face_membership()) |
                  tf::tag_normals(make_normals_view());
    auto point_normals_buff = tf::compute_point_normals(tagged);
    *_point_normals_array = make_numpy_array<nanobind::shape<-1, Dims>>(
        std::move(point_normals_buff.data_buffer()),
        {point_normals_buff.size(), Dims});
    }
    _point_normals_modified = false;
  }

  auto make_manifold_edge_link_view() {
    struct dref_t {
      auto operator()(Index i) const -> tf::manifold_edge_peer<Index> {
        return {i};
      }
    };
    const Index *offsets_data = static_cast<const Index *>(
        _manifold_edge_link_array->offsets_array().data());
    const Index *data_data = static_cast<const Index *>(
        _manifold_edge_link_array->data_array().data());
    auto offsets_range = tf::make_range(
        offsets_data, _manifold_edge_link_array->offsets_array().size());
    auto data_range = tf::make_range(
        data_data, _manifold_edge_link_array->data_array().size());
    // Map the data to manifold_edge_peer
    auto mapped = tf::make_mapped_range(data_range, dref_t{});
    auto mel = tf::make_offset_block_range(offsets_range, mapped);
    return tf::make_manifold_edge_link_like(mel);
  }

  auto make_normals_view() {
    const RealT *data = static_cast<const RealT *>(_normals_array->data());
    std::size_t count = _normals_array->shape(0) * Dims;
    return tf::make_unit_vectors<Dims>(tf::make_range(data, count));
  }

  auto make_point_normals_view() {
    const RealT *data =
        static_cast<const RealT *>(_point_normals_array->data());
    std::size_t count = _point_normals_array->shape(0) * Dims;
    return tf::make_unit_vectors<Dims>(tf::make_range(data, count));
  }

  std::unique_ptr<tf::py::offset_blocked_array_wrapper<Index, Index>>
      _faces_array;
  nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
      _points_array;
  std::unique_ptr<tf::aabb_tree<Index, RealT, Dims>> _tree;
  std::unique_ptr<tf::py::offset_blocked_array_wrapper<Index, Index>>
      _face_membership_array;
  std::unique_ptr<tf::py::offset_blocked_array_wrapper<Index, Index>>
      _manifold_edge_link_array;
  std::unique_ptr<tf::py::offset_blocked_array_wrapper<Index, Index>>
      _face_link_array;
  std::unique_ptr<tf::py::offset_blocked_array_wrapper<Index, Index>>
      _vertex_link_array;
  std::unique_ptr<
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>>
      _normals_array;
  std::unique_ptr<
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>>
      _point_normals_array;
  bool _tree_modified = false;
  bool _face_membership_modified = false;
  bool _manifold_edge_link_modified = false;
  bool _face_link_modified = false;
  bool _vertex_link_modified = false;
  bool _normals_modified = false;
  bool _point_normals_modified = false;
};

} // namespace tf::py
