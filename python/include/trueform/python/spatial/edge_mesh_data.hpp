/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/offset_blocked_array.hpp"
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/segments.hpp>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/spatial/tree_config.hpp>
#include <trueform/topology/edge_membership.hpp>
#include <trueform/topology/vertex_link.hpp>

namespace tf::py {

template <typename Index, typename RealT, std::size_t Dims>
class edge_mesh_data_wrapper {
public:
  edge_mesh_data_wrapper() = default;

  edge_mesh_data_wrapper(
      nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, 2>>
          edges_array,
      nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
          points_array)
      : _edges_array{edges_array}, _points_array{points_array} {}

  // Create view into Python-owned array
  auto make_primitive_range() {
    RealT *data_pts = static_cast<RealT *>(_points_array.data());
    std::size_t count_pts = _points_array.shape(0) * Dims;
    auto pts = tf::make_points<Dims>(tf::make_range(data_pts, count_pts));
    Index *data_fcs = static_cast<Index *>(_edges_array.data());
    std::size_t count_fcs = _edges_array.shape(0) * 2;
    auto edges = tf::make_blocked_range<2>(tf::make_range(data_fcs, count_fcs));
    return tf::make_segments(edges, pts);
  }

  auto make_primitive_range() const {
    const RealT *data_pts = static_cast<const RealT *>(_points_array.data());
    std::size_t count_pts = _points_array.shape(0) * Dims;
    auto pts = tf::make_points<Dims>(tf::make_range(data_pts, count_pts));
    const Index *data_fcs = static_cast<const Index *>(_edges_array.data());
    std::size_t count_fcs = _edges_array.shape(0) * 2;
    auto edges = tf::make_blocked_range<2>(tf::make_range(data_fcs, count_fcs));
    return tf::make_segments(edges, pts);
  }

  // Tree management
  auto rebuild_tree() -> void {
    if (!_tree) {
      _tree = std::make_unique<tf::aabb_tree<Index, RealT, Dims>>();
    }
    auto polys = make_primitive_range();
    *_tree = tf::aabb_tree<Index, RealT, Dims>(polys, tf::config_tree(4, 4));
    _tree_modified = false;
  }

  auto ensure_tree() -> void {
    if (!_tree || _tree_modified) {
      rebuild_tree();
      _tree_modified = false;
    }
  }

  auto rebuild_vertex_link() -> void {
    auto segments = make_primitive_range();
    tf::vertex_link<Index> fm;
    fm.build(segments.edges(), Index(segments.points().size()));

    auto [offsets, data] = make_numpy_array(std::move(fm));

    if (!_vertex_link_array) {
      _vertex_link_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              offsets, data);
    } else {
      _vertex_link_array->set_arrays(offsets, data);
    }
    _vertex_link_modified = false;
  }

  auto ensure_vertex_link() -> void {
    if (!_vertex_link_array || _vertex_link_modified) {
      rebuild_vertex_link();
      _vertex_link_modified = false;
    }
  }

  auto vertex_link() const {
    return tf::make_vertex_link_like(_vertex_link_array->make_range());
  }

  auto clear_vertex_link() -> void { _vertex_link_array.reset(); }

  auto has_vertex_link() const -> bool { return _vertex_link_array != nullptr; }

  auto vertex_link_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    ensure_vertex_link();
    return *_vertex_link_array;
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

  auto rebuild_edge_membership() -> void {
    auto segments = make_primitive_range();
    tf::edge_membership<Index> fm;
    fm.build(segments);

    auto [offsets, data] = make_numpy_array(std::move(fm));

    if (!_edge_membership_array) {
      _edge_membership_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              offsets, data);
    } else {
      _edge_membership_array->set_arrays(offsets, data);
    }
    _edge_membership_modified = false;
  }

  auto ensure_edge_membership() -> void {
    if (!_edge_membership_array || _edge_membership_modified) {
      rebuild_edge_membership();
      _edge_membership_modified = false;
    }
  }

  auto edge_membership() const {
    return _edge_membership_array->make_range();
  }

  auto mark_modified() -> void {
    _tree_modified = true;
    _edge_membership_modified = true;
    _vertex_link_modified = true;
  }

  auto clear_tree() -> void { _tree.reset(); }

  auto clear_edge_membership() -> void { _edge_membership_array.reset(); }

  auto has_tree() const -> bool { return _tree != nullptr; }

  auto has_edge_membership() const -> bool {
    return _edge_membership_array != nullptr;
  }

  auto number_of_edges() const -> std::size_t { return _edges_array.shape(0); }

  auto number_of_points() const -> std::size_t {
    return _points_array.shape(0);
  }

  auto dims() const -> std::size_t { return Dims; }

  auto tree() -> tf::aabb_tree<Index, RealT, Dims> & {
    ensure_tree();
    return *_tree;
  }

  auto tree() const -> const tf::aabb_tree<Index, RealT, Dims> & {
    if (!_tree)
      throw std::runtime_error("Tree not built");
    return *_tree;
  }

  auto edge_membership_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    ensure_edge_membership();
    return *_edge_membership_array;
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

  auto edges_array() const
      -> nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, 2>> {
    return _edges_array;
  }

  auto set_edges_array(
      nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, 2>>
          edges_array) -> void {
    _edges_array = edges_array;
    mark_modified();
  }

  auto
  set_edge_membership(tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    if (!_edge_membership_array)
      _edge_membership_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              fm.offsets_array(), fm.data_array());
    else
      _edge_membership_array->set_arrays(fm.offsets_array(), fm.data_array());
    _edge_membership_modified = false;
  }

private:
  nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, 2>>
      _edges_array;
  nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
      _points_array;
  std::unique_ptr<tf::aabb_tree<Index, RealT, Dims>> _tree;
  std::unique_ptr<tf::py::offset_blocked_array_wrapper<Index, Index>>
      _edge_membership_array;
  std::unique_ptr<tf::py::offset_blocked_array_wrapper<Index, Index>>
      _vertex_link_array;
  bool _tree_modified = false;
  bool _edge_membership_modified = false;
  bool _vertex_link_modified = false;
};

} // namespace tf::py
