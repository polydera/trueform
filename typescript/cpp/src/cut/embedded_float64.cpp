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

#include "./embedded_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_cut_embedded_float64) {
  using Real = double;
  using namespace tf::ts;

  // Result types
  emscripten::value_object<cut_result_t<Real>>("CutResultFloat64")
      .field("mesh", &cut_result_t<Real>::mesh)
      .field("faceLabels", &cut_result_t<Real>::face_labels);

  emscripten::value_object<cut_result_with_curves_t<Real>>(
      "CutResultWithCurvesFloat64")
      .field("mesh", &cut_result_with_curves_t<Real>::mesh)
      .field("faceLabels", &cut_result_with_curves_t<Real>::face_labels)
      .field("curves", &cut_result_with_curves_t<Real>::curves);

  // Embedded intersection curves — sync
  emscripten::function("embedded_intersection_curves_float64",
                       &sync_embedded_intersection_curves<Real>);
  emscripten::function("embedded_intersection_curves_with_curves_float64",
                       &sync_embedded_intersection_curves_with_curves<Real>);

  // Embedded intersection curves — async
  emscripten::function("dispatch_embedded_intersection_curves_float64",
                       &async_embedded_intersection_curves<Real>);
  emscripten::function(
      "dispatch_embedded_intersection_curves_with_curves_float64",
      &async_embedded_intersection_curves_with_curves<Real>);

  // Embedded self-intersection curves — sync
  emscripten::function("embedded_self_intersection_curves_float64",
                       &sync_embedded_self_intersection_curves<Real>);
  emscripten::function(
      "embedded_self_intersection_curves_with_curves_float64",
      &sync_embedded_self_intersection_curves_with_curves<Real>);

  // Embedded self-intersection curves — async
  emscripten::function("dispatch_embedded_self_intersection_curves_float64",
                       &async_embedded_self_intersection_curves<Real>);
  emscripten::function(
      "dispatch_embedded_self_intersection_curves_with_curves_float64",
      &async_embedded_self_intersection_curves_with_curves<Real>);
}
