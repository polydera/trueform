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

#include "trueform/ts/core/mesh_data.hpp"

#include "trueform/topology/face_link.hpp"
#include "trueform/topology/face_membership.hpp"
#include "trueform/topology/manifold_edge_link.hpp"
#include "trueform/topology/vertex_link.hpp"
#include "trueform/geometry/compute_normals.hpp"
#include "trueform/geometry/compute_point_normals.hpp"

namespace tf {
namespace ts {

// -- Spatial tree --

auto mesh_data::ensure_tree() -> void {
  if (_tree && _tree_faces_gen == _faces_gen &&
      _tree_points_gen == _points_gen)
    return;
  _tree = std::make_shared<tf::aabb_tree<int, float, 3>>(
      polygons_range(), tf::config_tree(4, 4));
  _tree_faces_gen = _faces_gen;
  _tree_points_gen = _points_gen;
}

// -- Half-edge access --

auto mesh_data::ensure_half_edges() -> void {
  if (_he && _he_gen == _faces_gen)
    return;
  ensure_face_membership();
  _he = std::make_shared<tf::half_edges<int>>();
  _he->build(faces_range(), face_membership_range());
  _he_gen = _faces_gen;
}

// -- Topology access --

auto mesh_data::face_membership() -> wasm_offset_blocked_buffer<int, int> {
  ensure_face_membership();
  return _fm;
}

auto mesh_data::manifold_edge_link() -> wasm_ndarray<int> {
  ensure_manifold_edge_link();
  return _mel;
}

auto mesh_data::face_link() -> wasm_offset_blocked_buffer<int, int> {
  ensure_face_link();
  return _fl;
}

auto mesh_data::vertex_link() -> wasm_offset_blocked_buffer<int, int> {
  ensure_vertex_link();
  return _vl;
}

// -- Normals access --

auto mesh_data::normals() -> wasm_ndarray<float> {
  ensure_normals();
  return _normals;
}

auto mesh_data::point_normals() -> wasm_ndarray<float> {
  ensure_point_normals();
  return _point_normals;
}

// -- Private build methods --

auto mesh_data::ensure_face_membership() -> void {
  if (_fm.is_valid() && _fm_gen == _faces_gen)
    return;
  tf::face_membership<int> fm;
  fm.build(polygons_range());
  _fm = wasm_offset_blocked_buffer<int, int>::from_buffer(
      std::move(static_cast<tf::offset_block_buffer<int, int> &>(fm)));
  _fm_gen = _faces_gen;
}

auto mesh_data::ensure_manifold_edge_link() -> void {
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

auto mesh_data::ensure_face_link() -> void {
  if (_fl.is_valid() && _fl_gen == _faces_gen)
    return;

  ensure_face_membership();

  tf::face_link<int> fl;
  fl.build(faces_range(), face_membership_range());

  _fl = wasm_offset_blocked_buffer<int, int>::from_buffer(
      std::move(static_cast<tf::offset_block_buffer<int, int> &>(fl)));
  _fl_gen = _faces_gen;
}

auto mesh_data::ensure_vertex_link() -> void {
  if (_vl.is_valid() && _vl_gen == _faces_gen)
    return;

  ensure_face_membership();

  tf::vertex_link<int> vl;
  vl.build(faces_range(), face_membership_range());

  _vl = wasm_offset_blocked_buffer<int, int>::from_buffer(
      std::move(static_cast<tf::offset_block_buffer<int, int> &>(vl)));
  _vl_gen = _faces_gen;
}

auto mesh_data::ensure_normals() -> void {
  if (_normals.is_valid() && _normals_faces_gen == _faces_gen &&
      _normals_points_gen == _points_gen)
    return;
  auto nrm = tf::compute_normals(polygons_range());
  _normals = wasm_ndarray<float>::from_buffer(std::move(nrm.data_buffer()),
                                              {number_of_faces(), 3});
  _normals_faces_gen = _faces_gen;
  _normals_points_gen = _points_gen;
}

auto mesh_data::ensure_point_normals() -> void {
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
