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
#include <nanobind/stl/optional.h>
#include <optional>
#include <trueform/core/points.hpp>
#include <trueform/core/polygons.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/transformation_view.hpp>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/spatial/tree_config.hpp>
#include <trueform/topology/face_link.hpp>
#include <trueform/topology/face_membership.hpp>
#include <trueform/topology/manifold_edge_link.hpp>
#include <trueform/topology/vertex_link.hpp>

namespace tf::py {

template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
class mesh_wrapper {
public:
  mesh_wrapper() = default;

  mesh_wrapper(
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

  // Tree management
  auto rebuild_tree() -> void {
    if (!_tree) {
      _tree = std::make_unique<tf::aabb_tree<Index, RealT, Dims>>();
    }
    auto polys = make_primitive_range();
    *_tree = tf::aabb_tree<Index, RealT, Dims>(polys, tf::config_tree(4, 4));
  }

  auto ensure_tree() -> void {
    if (!_tree) {
      rebuild_tree();
    }
  }

  auto rebuild_face_link() -> void {
    auto polygons = make_primitive_range();
    tf::face_link<Index> fm;
    fm.build(polygons.faces(), face_membership());

    auto [offsets, data] = make_numpy_array(std::move(fm));

    // Pass the numpy arrays to the wrapper
    if (!_face_link_array) {
      _face_link_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              offsets, data);
    } else {
      _face_link_array->set_arrays(offsets, data);
    }
  }

  auto ensure_face_link() -> void {
    ensure_face_membership();
    if (!_face_link_array) {
      rebuild_face_link();
    }
  }

  auto face_link() const {
    return tf::make_face_link_like(_face_link_array->make_range());
  }

  auto clear_face_link() -> void { _face_link_array.reset(); }

  auto has_face_link() const -> bool { return _face_link_array != nullptr; }

  auto face_link_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    ensure_face_link();
    return *_face_link_array;
  }

  auto set_face_link(tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    if (!_face_link_array)
      _face_link_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              fm.offsets_array(), fm.data_array());
    else
      _face_link_array->set_arrays(fm.offsets_array(), fm.data_array());
  }

  auto rebuild_vertex_link() -> void {
    auto polygons = make_primitive_range();
    tf::vertex_link<Index> fm;
    fm.build(polygons.faces(), face_membership());

    auto [offsets, data] = make_numpy_array(std::move(fm));

    // Pass the numpy arrays to the wrapper
    if (!_vertex_link_array) {
      _vertex_link_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              offsets, data);
    } else {
      _vertex_link_array->set_arrays(offsets, data);
    }
  }

  auto ensure_vertex_link() -> void {
    ensure_face_membership();
    if (!_vertex_link_array) {
      rebuild_vertex_link();
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
  }

  auto rebuild_face_membership() -> void {
    auto polygons = make_primitive_range();
    tf::face_membership<Index> fm;
    fm.build(polygons);

    auto [offsets, data] = make_numpy_array(std::move(fm));

    // Pass the numpy arrays to the wrapper
    if (!_face_membership_array) {
      _face_membership_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              offsets, data);
    } else {
      _face_membership_array->set_arrays(offsets, data);
    }
  }

  auto ensure_face_membership() -> void {
    if (!_face_membership_array) {
      rebuild_face_membership();
    }
  }

  auto face_membership() const {
    return tf::make_face_membership_like(_face_membership_array->make_range());
  }

  auto rebuild_manifold_edge_link() -> void {
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
  }

  auto ensure_manifold_edge_link() -> void {
    ensure_face_membership();
    if (!_manifold_edge_link_array) {
      rebuild_manifold_edge_link();
    }
  }

  auto manifold_edge_link() const {
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

  auto clear_tree() -> void { _tree.reset(); }

  auto clear_face_membership() -> void { _face_membership_array.reset(); }

  auto clear_manifold_edge_link() -> void { _manifold_edge_link_array.reset(); }

  auto has_tree() const -> bool { return _tree != nullptr; }

  auto has_face_membership() const -> bool {
    return _face_membership_array != nullptr;
  }

  auto has_manifold_edge_link() const -> bool {
    return _manifold_edge_link_array != nullptr;
  }

  auto number_of_faces() const -> std::size_t { return _faces_array.shape(0); }

  auto number_of_points() const -> std::size_t {
    return _points_array.shape(0);
  }

  auto dims() const -> std::size_t { return Dims; }

  // Access to internal structures (opaque to Python)
  auto tree() -> tf::aabb_tree<Index, RealT, Dims> & {
    ensure_tree();
    return *_tree;
  }

  auto tree() const -> const tf::aabb_tree<Index, RealT, Dims> & {
    if (!_tree)
      throw std::runtime_error("Tree not built");
    return *_tree;
  }

  auto face_membership_array()
      -> const tf::py::offset_blocked_array_wrapper<Index, Index> & {
    ensure_face_membership();
    return *_face_membership_array;
  }

  auto manifold_edge_link_array() -> const
      nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>> & {
    ensure_manifold_edge_link();
    return *_manifold_edge_link_array;
  }

  auto points_array() const
      -> nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>> {
    return _points_array;
  }

  auto faces_array() const
      -> nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>> {
    return _faces_array;
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

  auto
  set_face_membership(tf::py::offset_blocked_array_wrapper<Index, Index> fm) {
    if (!_face_membership_array)
      _face_membership_array =
          std::make_unique<tf::py::offset_blocked_array_wrapper<Index, Index>>(
              fm.offsets_array(), fm.data_array());
    else
      _face_membership_array->set_arrays(fm.offsets_array(), fm.data_array());
  }

  auto set_manifold_edge_link(
      nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>>
          mel) {
    if (!_manifold_edge_link_array)
      _manifold_edge_link_array =
          std::make_unique<nanobind::ndarray<nanobind::numpy, Index,
                                             nanobind::shape<-1, Ngon>>>();
    *_manifold_edge_link_array = mel;
  }

  auto clear_transformation() -> void { _transformation.reset(); }

private:
  nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>>
      _faces_array;
  nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, Dims>>
      _points_array;
  std::optional<nanobind::ndarray<nanobind::numpy, RealT,
                                  nanobind::shape<Dims + 1, Dims + 1>>>
      _transformation;
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
};

} // namespace tf::py
