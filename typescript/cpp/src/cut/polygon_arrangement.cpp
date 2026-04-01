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

#include "trueform/cut/make_polygon_arrangements.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/cut/result_types.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

struct polygon_arrangement_result {
  wasm_mesh mesh;
  wasm_ndarray<int> face_labels;
};

struct polygon_arrangement_result_with_curves {
  wasm_mesh mesh;
  wasm_ndarray<int> face_labels;
  wasm_curves curves;
};

auto sync_polygon_arrangements(wasm_mesh &m, int mode)
    -> polygon_arrangement_result {
  auto fm = m.face_membership_range();
  auto mel = m.manifold_edge_link_range();
  auto [mesh, face_labels] = tf::make_polygon_arrangements(
      m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) | tf::tag(mel),
      static_cast<tf::intersect_mode>(mode));
  return {wasm_mesh::from_polygons_buffer(std::move(mesh)),
          wasm_ndarray<int>::from_buffer(std::move(face_labels))};
}

auto sync_polygon_arrangements_with_curves(wasm_mesh &m, int mode)
    -> polygon_arrangement_result_with_curves {
  auto fm = m.face_membership_range();
  auto mel = m.manifold_edge_link_range();
  auto [mesh, face_labels, curves] = tf::make_polygon_arrangements(
      m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) | tf::tag(mel),
      static_cast<tf::intersect_mode>(mode), tf::return_curves);
  return {wasm_mesh::from_polygons_buffer(std::move(mesh)),
          wasm_ndarray<int>::from_buffer(std::move(face_labels)),
          wasm_curves::from_curves_buffer(std::move(curves))};
}

auto async_polygon_arrangements(wasm_mesh &m, int mode) -> promise_t {
  return promise([a = m, mode]() -> polygon_arrangement_result {
    return sync_polygon_arrangements(const_cast<wasm_mesh &>(a), mode);
  });
}

auto async_polygon_arrangements_with_curves(wasm_mesh &m, int mode)
    -> promise_t {
  return promise([a = m, mode]() -> polygon_arrangement_result_with_curves {
    return sync_polygon_arrangements_with_curves(const_cast<wasm_mesh &>(a),
                                                 mode);
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_polygon_arrangement) {
  emscripten::value_object<polygon_arrangement_result>("polygon_arrangement_result")
      .field("mesh", &polygon_arrangement_result::mesh)
      .field("faceLabels", &polygon_arrangement_result::face_labels);

  emscripten::value_object<polygon_arrangement_result_with_curves>(
      "polygon_arrangement_result_with_curves")
      .field("mesh", &polygon_arrangement_result_with_curves::mesh)
      .field("faceLabels", &polygon_arrangement_result_with_curves::face_labels)
      .field("curves", &polygon_arrangement_result_with_curves::curves);

  emscripten::function("polygon_arrangements", &sync_polygon_arrangements);
  emscripten::function("dispatch_polygon_arrangements",
                       &async_polygon_arrangements);
  emscripten::function("polygon_arrangements_with_curves",
                       &sync_polygon_arrangements_with_curves);
  emscripten::function("dispatch_polygon_arrangements_with_curves",
                       &async_polygon_arrangements_with_curves);
}
