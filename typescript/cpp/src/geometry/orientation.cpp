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

#include "trueform/core/algorithm/parallel_copy.hpp"
#include "trueform/geometry/ensure_positive_orientation.hpp"
#include "trueform/topology/reverse_winding.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

auto sync_positively_oriented(wasm_mesh &m, bool is_consistent) -> wasm_mesh {
  const auto num_faces = m.number_of_faces();
  const auto num_ints = static_cast<std::size_t>(num_faces) * 3;

  // Copy faces into a new mutable buffer
  tf::buffer<int> face_buf;
  face_buf.allocate(num_ints);
  auto src = m.faces().make_range();
  tf::parallel_copy(src, tf::make_range(face_buf.data(), num_ints));

  // Build mutable polygons from the copy + original points
  auto faces = tf::make_faces<3>(tf::make_range(face_buf.data(), num_ints));
  auto points = m.points_range();
  auto polys = tf::make_polygons(faces, points);

  // Tag with cached manifold edge link from the original mesh
  auto mel = m.manifold_edge_link_range();
  auto tagged = polys | tf::tag(mel);

  tf::ensure_positive_orientation(tagged, is_consistent);

  // New faces ndarray from modified buffer; share points from original
  auto result_faces =
      wasm_ndarray<int>::from_buffer(std::move(face_buf), {num_faces, 3});
  return wasm_mesh::create(std::move(result_faces), m.points());
}

auto async_positively_oriented(wasm_mesh &m, bool is_consistent) -> promise_t {
  return promise([a = m, is_consistent]() -> wasm_mesh {
    return sync_positively_oriented(const_cast<wasm_mesh &>(a), is_consistent);
  });
}

auto sync_reverse_winding(wasm_mesh &m) -> wasm_mesh {
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
  return wasm_mesh::create(std::move(result_faces), m.points());
}

auto async_reverse_winding(wasm_mesh &m) -> promise_t {
  return promise([a = m]() -> wasm_mesh {
    return sync_reverse_winding(const_cast<wasm_mesh &>(a));
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_geometry_orientation) {
  emscripten::function("positively_oriented", &sync_positively_oriented);
  emscripten::function("dispatch_positively_oriented",
                       &async_positively_oriented);
  emscripten::function("reverse_winding", &sync_reverse_winding);
  emscripten::function("dispatch_reverse_winding", &async_reverse_winding);
}
