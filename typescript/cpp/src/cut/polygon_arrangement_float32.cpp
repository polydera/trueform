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

#include "./polygon_arrangement_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_cut_polygon_arrangement_float32) {
  using Real = float;
  using namespace tf::ts;

  // Result types
  emscripten::value_object<polygon_arrangement_result_t<Real>>(
      "PolygonArrangementResultFloat32")
      .field("mesh", &polygon_arrangement_result_t<Real>::mesh)
      .field("faceLabels", &polygon_arrangement_result_t<Real>::face_labels);

  emscripten::value_object<polygon_arrangement_result_with_curves_t<Real>>(
      "PolygonArrangementResultWithCurvesFloat32")
      .field("mesh", &polygon_arrangement_result_with_curves_t<Real>::mesh)
      .field("faceLabels",
             &polygon_arrangement_result_with_curves_t<Real>::face_labels)
      .field("curves",
             &polygon_arrangement_result_with_curves_t<Real>::curves);

  // Sync
  emscripten::function("polygon_arrangements_float32",
                       &sync_polygon_arrangements<Real>);
  emscripten::function("polygon_arrangements_with_curves_float32",
                       &sync_polygon_arrangements_with_curves<Real>);

  // Async
  emscripten::function("dispatch_polygon_arrangements_float32",
                       &async_polygon_arrangements<Real>);
  emscripten::function("dispatch_polygon_arrangements_with_curves_float32",
                       &async_polygon_arrangements_with_curves<Real>);
}
