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

#include "trueform/ts/core/wasm_mesh.hpp"

#include "trueform/topology/face_link.hpp"
#include "trueform/topology/face_membership.hpp"
#include "trueform/topology/manifold_edge_link.hpp"
#include "trueform/topology/vertex_link.hpp"
#include "trueform/geometry/compute_normals.hpp"
#include "trueform/geometry/compute_point_normals.hpp"
#include <emscripten/bind.h>

namespace tf {
namespace ts {

// (create / set_faces / set_points are now inline in wasm_mesh.hpp)

// -- Spatial tree --

void wasm_mesh::ensure_tree() {
  if (_tree && _tree_faces_gen == _faces_gen &&
      _tree_points_gen == _points_gen)
    return;
  _tree = std::make_shared<tf::aabb_tree<int, float, 3>>(
      polygons_range(), tf::config_tree(4, 4));
  _tree_faces_gen = _faces_gen;
  _tree_points_gen = _points_gen;
}

// -- Half-edge access --

auto wasm_mesh::ensure_half_edges() -> void {
  if (_he && _he_gen == _faces_gen)
    return;
  ensure_face_membership();
  _he = std::make_shared<tf::half_edges<int>>();
  _he->build(faces_range(), face_membership_range());
  _he_gen = _faces_gen;
}

// -- Topology access --

auto wasm_mesh::face_membership() -> wasm_offset_blocked_buffer<int, int> {
  ensure_face_membership();
  return _fm;
}

auto wasm_mesh::manifold_edge_link() -> wasm_ndarray<int> {
  ensure_manifold_edge_link();
  return _mel;
}

auto wasm_mesh::face_link() -> wasm_offset_blocked_buffer<int, int> {
  ensure_face_link();
  return _fl;
}

auto wasm_mesh::vertex_link() -> wasm_offset_blocked_buffer<int, int> {
  ensure_vertex_link();
  return _vl;
}

// -- Normals access --

auto wasm_mesh::normals() -> wasm_ndarray<float> {
  ensure_normals();
  return _normals;
}

auto wasm_mesh::point_normals() -> wasm_ndarray<float> {
  ensure_point_normals();
  return _point_normals;
}

// -- Private build methods --

void wasm_mesh::ensure_face_membership() {
  if (_fm.is_valid() && _fm_gen == _faces_gen)
    return;
  tf::face_membership<int> fm;
  fm.build(polygons_range());
  _fm = wasm_offset_blocked_buffer<int, int>::from_buffer(
      std::move(static_cast<tf::offset_block_buffer<int, int> &>(fm)));
  _fm_gen = _faces_gen;
}

void wasm_mesh::ensure_manifold_edge_link() {
  if (_mel.is_valid() && _mel_gen == _faces_gen)
    return;

  ensure_face_membership();

  tf::manifold_edge_link<int, 3> mel;
  mel.build(faces_range(), face_membership_range());

  // Extract face_peer values into flat int buffer [F, 3]
  auto &buf = mel.data_buffer();
  auto total = buf.size();
  tf::buffer<int> out;
  out.allocate(total);
  for (std::size_t i = 0; i < total; ++i)
    out[i] = buf[i].face_peer;

  _mel = wasm_ndarray<int>::from_buffer(std::move(out), {number_of_faces(), 3});
  _mel_gen = _faces_gen;
}

void wasm_mesh::ensure_face_link() {
  if (_fl.is_valid() && _fl_gen == _faces_gen)
    return;

  ensure_face_membership();

  tf::face_link<int> fl;
  fl.build(faces_range(), face_membership_range());

  _fl = wasm_offset_blocked_buffer<int, int>::from_buffer(
      std::move(static_cast<tf::offset_block_buffer<int, int> &>(fl)));
  _fl_gen = _faces_gen;
}

void wasm_mesh::ensure_vertex_link() {
  if (_vl.is_valid() && _vl_gen == _faces_gen)
    return;

  ensure_face_membership();

  tf::vertex_link<int> vl;
  vl.build(faces_range(), face_membership_range());

  _vl = wasm_offset_blocked_buffer<int, int>::from_buffer(
      std::move(static_cast<tf::offset_block_buffer<int, int> &>(vl)));
  _vl_gen = _faces_gen;
}

void wasm_mesh::ensure_normals() {
  if (_normals.is_valid() && _normals_faces_gen == _faces_gen &&
      _normals_points_gen == _points_gen)
    return;
  auto nrm = tf::compute_normals(polygons_range());
  _normals = wasm_ndarray<float>::from_buffer(std::move(nrm.data_buffer()),
                                              {number_of_faces(), 3});
  _normals_faces_gen = _faces_gen;
  _normals_points_gen = _points_gen;
}

void wasm_mesh::ensure_point_normals() {
  if (_point_normals.is_valid() && _point_normals_faces_gen == _faces_gen &&
      _point_normals_points_gen == _points_gen)
    return;

  ensure_normals();
  ensure_face_membership();

  // Reuse cached face normals and face membership
  auto normals_view = tf::make_unit_vectors<3>(_normals.make_range());
  auto fm = face_membership_range();
  auto tagged =
      polygons_range() | tf::tag(fm) | tf::tag_normals(normals_view);

  auto pn = tf::compute_point_normals(tagged);
  _point_normals = wasm_ndarray<float>::from_buffer(
      std::move(pn.data_buffer()), {number_of_points(), 3});
  _point_normals_faces_gen = _faces_gen;
  _point_normals_points_gen = _points_gen;
}

} // namespace ts
} // namespace tf

EMSCRIPTEN_BINDINGS(trueform_mesh) {
  emscripten::class_<tf::ts::wasm_mesh>("NativeMesh")
      .class_function("create", &tf::ts::wasm_mesh::create)
      .function("faces", &tf::ts::wasm_mesh::faces)
      .function("points", &tf::ts::wasm_mesh::points)
      .function("number_of_faces", &tf::ts::wasm_mesh::number_of_faces)
      .function("number_of_points", &tf::ts::wasm_mesh::number_of_points)
      .function("set_faces", &tf::ts::wasm_mesh::set_faces)
      .function("set_points", &tf::ts::wasm_mesh::set_points)
      .function("shared_view", &tf::ts::wasm_mesh::shared_view)
      .function("face_membership", &tf::ts::wasm_mesh::face_membership)
      .function("manifold_edge_link", &tf::ts::wasm_mesh::manifold_edge_link)
      .function("face_link", &tf::ts::wasm_mesh::face_link)
      .function("vertex_link", &tf::ts::wasm_mesh::vertex_link)
      .function("normals", &tf::ts::wasm_mesh::normals)
      .function("point_normals", &tf::ts::wasm_mesh::point_normals)
      .function("set_face_membership", &tf::ts::wasm_mesh::set_face_membership)
      .function("set_vertex_link", &tf::ts::wasm_mesh::set_vertex_link)
      .function("set_face_link", &tf::ts::wasm_mesh::set_face_link)
      .function("set_manifold_edge_link", &tf::ts::wasm_mesh::set_manifold_edge_link)
      .function("set_normals", &tf::ts::wasm_mesh::set_normals)
      .function("set_point_normals", &tf::ts::wasm_mesh::set_point_normals)
      .function("has_transformation", &tf::ts::wasm_mesh::has_transformation)
      .function("transformation", &tf::ts::wasm_mesh::transformation)
      .function("set_transformation", &tf::ts::wasm_mesh::set_transformation)
      .function("clear_transformation", &tf::ts::wasm_mesh::clear_transformation)
      .function("ensure_tree", &tf::ts::wasm_mesh::ensure_tree)
      .function("destroy", &tf::ts::wasm_mesh::destroy)
      .function("is_valid", &tf::ts::wasm_mesh::is_valid);
}
