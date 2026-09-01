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

#include "./isobands_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_iso_isobands_float64) {
  using Real = double;
  using namespace tf::ts;

  // Result types
  emscripten::value_object<isobands_result_t<Real>>("IsobandsResultFloat64")
      .field("mesh", &isobands_result_t<Real>::mesh)
      .field("labels", &isobands_result_t<Real>::labels)
      .field("faceLabels", &isobands_result_t<Real>::face_labels);

  emscripten::value_object<isobands_result_with_curves_t<Real>>(
      "IsobandsResultWithCurvesFloat64")
      .field("mesh", &isobands_result_with_curves_t<Real>::mesh)
      .field("labels", &isobands_result_with_curves_t<Real>::labels)
      .field("faceLabels", &isobands_result_with_curves_t<Real>::face_labels)
      .field("curves", &isobands_result_with_curves_t<Real>::curves);

  // Sync
  emscripten::function("isobands_float64", &sync_isobands<Real>);
  emscripten::function("isobands_with_curves_float64",
                       &sync_isobands_with_curves<Real>);
  emscripten::function("isobands_selected_float64",
                       &sync_isobands_selected<Real>);
  emscripten::function("isobands_with_curves_selected_float64",
                       &sync_isobands_with_curves_selected<Real>);

  // Async
  emscripten::function("dispatch_isobands_float64", &async_isobands<Real>);
  emscripten::function("dispatch_isobands_with_curves_float64",
                       &async_isobands_with_curves<Real>);
  emscripten::function("dispatch_isobands_selected_float64",
                       &async_isobands_selected<Real>);
  emscripten::function("dispatch_isobands_with_curves_selected_float64",
                       &async_isobands_with_curves_selected<Real>);
}
