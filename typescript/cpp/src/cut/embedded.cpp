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

auto sync_embedded_intersection_curves(wasm_mesh &m0, wasm_mesh &m1)
    -> wasm_mesh {
  auto poly = tf::embedded_intersection_curves(m0.polygons_range(),
                                               m1.polygons_range());
  return wasm_mesh::from_polygons_buffer(std::move(poly));
}

auto sync_embedded_intersection_curves_with_curves(wasm_mesh &m0,
                                                   wasm_mesh &m1)
    -> cut_result_with_curves {
  auto [poly, curves] = tf::embedded_intersection_curves(
      m0.polygons_range(), m1.polygons_range(), tf::return_curves);
  return {wasm_mesh::from_polygons_buffer(std::move(poly)),
          wasm_curves::from_curves_buffer(std::move(curves))};
}

auto async_embedded_intersection_curves(wasm_mesh &m0, wasm_mesh &m1)
    -> promise_t {
  return promise([a = m0, b = m1]() -> wasm_mesh {
    return sync_embedded_intersection_curves(const_cast<wasm_mesh &>(a),
                                             const_cast<wasm_mesh &>(b));
  });
}

auto async_embedded_intersection_curves_with_curves(wasm_mesh &m0,
                                                    wasm_mesh &m1)
    -> promise_t {
  return promise([a = m0, b = m1]() -> cut_result_with_curves {
    return sync_embedded_intersection_curves_with_curves(
        const_cast<wasm_mesh &>(a), const_cast<wasm_mesh &>(b));
  });
}

// ============================================================================
// Embedded self-intersection curves (mesh → split mesh)
// ============================================================================

auto sync_embedded_self_intersection_curves(wasm_mesh &m) -> wasm_mesh {
  auto poly = tf::embedded_self_intersection_curves(m.polygons_range());
  return wasm_mesh::from_polygons_buffer(std::move(poly));
}

auto sync_embedded_self_intersection_curves_with_curves(wasm_mesh &m)
    -> cut_result_with_curves {
  auto [poly, curves] = tf::embedded_self_intersection_curves(
      m.polygons_range(), tf::return_curves);
  return {wasm_mesh::from_polygons_buffer(std::move(poly)),
          wasm_curves::from_curves_buffer(std::move(curves))};
}

auto async_embedded_self_intersection_curves(wasm_mesh &m) -> promise_t {
  return promise([a = m]() -> wasm_mesh {
    return sync_embedded_self_intersection_curves(const_cast<wasm_mesh &>(a));
  });
}

auto async_embedded_self_intersection_curves_with_curves(wasm_mesh &m)
    -> promise_t {
  return promise([a = m]() -> cut_result_with_curves {
    return sync_embedded_self_intersection_curves_with_curves(
        const_cast<wasm_mesh &>(a));
  });
}

} // namespace

EMSCRIPTEN_BINDINGS(trueform_embedded) {
  // Result type
  emscripten::value_object<tf::ts::cut_result_with_curves>(
      "CutResultWithCurves")
      .field("mesh", &tf::ts::cut_result_with_curves::mesh)
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
