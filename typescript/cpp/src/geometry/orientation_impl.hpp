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

#include "trueform/core/algorithm/parallel_copy.hpp"
#include "trueform/geometry/ensure_positive_orientation.hpp"
#include "trueform/topology/reverse_winding.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"

namespace tf {
namespace ts {

template <typename Real>
auto sync_positively_oriented(wasm_mesh<Real> &m, bool is_consistent)
    -> wasm_mesh<Real> {
  const auto num_faces = m.number_of_faces();
  const auto num_ints = static_cast<std::size_t>(num_faces) * 3;

  tf::buffer<int> face_buf;
  face_buf.allocate(num_ints);
  auto src = m.faces().make_range();
  tf::parallel_copy(src, tf::make_range(face_buf.data(), num_ints));

  auto faces = tf::make_faces<3>(tf::make_range(face_buf.data(), num_ints));
  auto points = m.points_range();
  auto polys = tf::make_polygons(faces, points);

  auto mel = m.manifold_edge_link_range();
  auto tagged = polys | tf::tag(mel);

  tf::ensure_positive_orientation(tagged, is_consistent);

  auto result_faces =
      wasm_ndarray<int>::from_buffer(std::move(face_buf), {num_faces, 3});
  return wasm_mesh<Real>::create(std::move(result_faces), m.points());
}

template <typename Real>
auto async_positively_oriented(wasm_mesh<Real> &m, bool is_consistent)
    -> promise_t {
  return promise([a = m, is_consistent]() -> wasm_mesh<Real> {
    return sync_positively_oriented<Real>(const_cast<wasm_mesh<Real> &>(a),
                                          is_consistent);
  });
}

template <typename Real>
auto sync_reverse_winding(wasm_mesh<Real> &m) -> wasm_mesh<Real> {
  const auto num_faces = m.number_of_faces();
  const auto num_ints = static_cast<std::size_t>(num_faces) * 3;

  tf::buffer<int> face_buf;
  face_buf.allocate(num_ints);
  auto src = m.faces().make_range();
  tf::parallel_copy(src, tf::make_range(face_buf.data(), num_ints));

  auto faces = tf::make_faces<3>(tf::make_range(face_buf.data(), num_ints));
  tf::reverse_winding(faces);

  auto result_faces =
      wasm_ndarray<int>::from_buffer(std::move(face_buf), {num_faces, 3});
  return wasm_mesh<Real>::create(std::move(result_faces), m.points());
}

template <typename Real>
auto async_reverse_winding(wasm_mesh<Real> &m) -> promise_t {
  return promise([a = m]() -> wasm_mesh<Real> {
    return sync_reverse_winding<Real>(const_cast<wasm_mesh<Real> &>(a));
  });
}

} // namespace ts
} // namespace tf
