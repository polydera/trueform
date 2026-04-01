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

#include "trueform/cut/embedded_intersection_curves.hpp"
#include "trueform/cut/embedded_self_intersection_curves.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/cut/result_types.hpp"
#include <emscripten/bind.h>

namespace {

using namespace tf::ts;

// ============================================================================
// Embedded intersection curves (mesh0 × mesh1 → split mesh0)
// ============================================================================

auto sync_embedded_intersection_curves(wasm_mesh &m0, wasm_mesh &m1, int mode)
    -> cut_result {
  bool has0 = m0.has_transformation();
  bool has1 = m1.has_transformation();
  auto fm0 = m0.face_membership_range();
  auto mel0 = m0.manifold_edge_link_range();
  auto fm1 = m1.face_membership_range();
  auto mel1 = m1.manifold_edge_link_range();
  auto m = static_cast<tf::intersect_mode>(mode);
  auto make_return = [&](auto &&poly0, auto &&poly1) -> cut_result {
    auto [poly, face_labels] = tf::embedded_intersection_curves(
        poly0 | tf::tag(m0.tree()) | tf::tag(fm0) | tf::tag(mel0),
        poly1 | tf::tag(m1.tree()) | tf::tag(fm1) | tf::tag(mel1), m);
    return {wasm_mesh::from_polygons_buffer(std::move(poly)),
            wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels))};
  };
  if (has0 && has1)
    return make_return(
        m0.polygons_range() | tf::tag(m0.transformation_view()),
        m1.polygons_range() | tf::tag(m1.transformation_view()));
  else if (has0)
    return make_return(
        m0.polygons_range() | tf::tag(m0.transformation_view()),
        m1.polygons_range());
  else if (has1)
    return make_return(m0.polygons_range(),
                       m1.polygons_range() | tf::tag(m1.transformation_view()));
  else
    return make_return(m0.polygons_range(), m1.polygons_range());
}

auto sync_embedded_intersection_curves_with_curves(wasm_mesh &m0,
                                                   wasm_mesh &m1, int mode)
    -> cut_result_with_curves {
  bool has0 = m0.has_transformation();
  bool has1 = m1.has_transformation();
  auto fm0 = m0.face_membership_range();
  auto mel0 = m0.manifold_edge_link_range();
  auto fm1 = m1.face_membership_range();
  auto mel1 = m1.manifold_edge_link_range();
  auto m = static_cast<tf::intersect_mode>(mode);
  auto make_return = [&](auto &&poly0, auto &&poly1) -> cut_result_with_curves {
    auto [poly, face_labels, curves] = tf::embedded_intersection_curves(
        poly0 | tf::tag(m0.tree()) | tf::tag(fm0) | tf::tag(mel0),
        poly1 | tf::tag(m1.tree()) | tf::tag(fm1) | tf::tag(mel1),
        m, tf::return_curves);
    return {wasm_mesh::from_polygons_buffer(std::move(poly)),
            wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels)),
            wasm_curves::from_curves_buffer(std::move(curves))};
  };
  if (has0 && has1)
    return make_return(
        m0.polygons_range() | tf::tag(m0.transformation_view()),
        m1.polygons_range() | tf::tag(m1.transformation_view()));
  else if (has0)
    return make_return(
        m0.polygons_range() | tf::tag(m0.transformation_view()),
        m1.polygons_range());
  else if (has1)
    return make_return(m0.polygons_range(),
                       m1.polygons_range() | tf::tag(m1.transformation_view()));
  else
    return make_return(m0.polygons_range(), m1.polygons_range());
}

auto async_embedded_intersection_curves(wasm_mesh &m0, wasm_mesh &m1, int mode)
    -> promise_t {
  return promise([a = m0, b = m1, mode]() -> cut_result {
    return sync_embedded_intersection_curves(const_cast<wasm_mesh &>(a),
                                             const_cast<wasm_mesh &>(b), mode);
  });
}

auto async_embedded_intersection_curves_with_curves(wasm_mesh &m0,
                                                    wasm_mesh &m1, int mode)
    -> promise_t {
  return promise([a = m0, b = m1, mode]() -> cut_result_with_curves {
    return sync_embedded_intersection_curves_with_curves(
        const_cast<wasm_mesh &>(a), const_cast<wasm_mesh &>(b), mode);
  });
}

// ============================================================================
// Embedded self-intersection curves (mesh → split mesh)
// ============================================================================

auto sync_embedded_self_intersection_curves(wasm_mesh &m, int mode)
    -> cut_result {
  auto fm = m.face_membership_range();
  auto mel = m.manifold_edge_link_range();
  auto [poly, face_labels] = tf::embedded_self_intersection_curves(
      m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) | tf::tag(mel),
      static_cast<tf::intersect_mode>(mode));
  return {wasm_mesh::from_polygons_buffer(std::move(poly)),
          wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels))};
}

auto sync_embedded_self_intersection_curves_with_curves(wasm_mesh &m, int mode)
    -> cut_result_with_curves {
  auto fm = m.face_membership_range();
  auto mel = m.manifold_edge_link_range();
  auto [poly, face_labels, curves] = tf::embedded_self_intersection_curves(
      m.polygons_range() | tf::tag(m.tree()) | tf::tag(fm) | tf::tag(mel),
      static_cast<tf::intersect_mode>(mode), tf::return_curves);
  return {wasm_mesh::from_polygons_buffer(std::move(poly)),
          wasm_ndarray<std::int32_t>::from_buffer(std::move(face_labels)),
          wasm_curves::from_curves_buffer(std::move(curves))};
}

auto async_embedded_self_intersection_curves(wasm_mesh &m, int mode)
    -> promise_t {
  return promise([a = m, mode]() -> cut_result {
    return sync_embedded_self_intersection_curves(const_cast<wasm_mesh &>(a),
                                                  mode);
  });
}

auto async_embedded_self_intersection_curves_with_curves(wasm_mesh &m, int mode)
    -> promise_t {
  return promise([a = m, mode]() -> cut_result_with_curves {
    return sync_embedded_self_intersection_curves_with_curves(
        const_cast<wasm_mesh &>(a), mode);
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_embedded) {
  // Result types
  emscripten::value_object<tf::ts::cut_result>("CutResult")
      .field("mesh", &tf::ts::cut_result::mesh)
      .field("faceLabels", &tf::ts::cut_result::face_labels);

  emscripten::value_object<tf::ts::cut_result_with_curves>(
      "CutResultWithCurves")
      .field("mesh", &tf::ts::cut_result_with_curves::mesh)
      .field("faceLabels", &tf::ts::cut_result_with_curves::face_labels)
      .field("curves", &tf::ts::cut_result_with_curves::curves);

  // Embedded intersection curves — sync
  emscripten::function("embedded_intersection_curves",
                       &sync_embedded_intersection_curves);
  emscripten::function("embedded_intersection_curves_with_curves",
                       &sync_embedded_intersection_curves_with_curves);

  // Embedded intersection curves — async
  emscripten::function("dispatch_embedded_intersection_curves",
                       &async_embedded_intersection_curves);
  emscripten::function("dispatch_embedded_intersection_curves_with_curves",
                       &async_embedded_intersection_curves_with_curves);

  // Embedded self-intersection curves — sync
  emscripten::function("embedded_self_intersection_curves",
                       &sync_embedded_self_intersection_curves);
  emscripten::function("embedded_self_intersection_curves_with_curves",
                       &sync_embedded_self_intersection_curves_with_curves);

  // Embedded self-intersection curves — async
  emscripten::function("dispatch_embedded_self_intersection_curves",
                       &async_embedded_self_intersection_curves);
  emscripten::function("dispatch_embedded_self_intersection_curves_with_curves",
                       &async_embedded_self_intersection_curves_with_curves);
}
