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

#include "./mesh_arrangement_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_arrangement_mesh_arrangement_float64) {
  using Real = double;
  using namespace tf::ts;

  // Result types
  emscripten::value_object<arrangement_result_t<Real>>(
      "MeshArrangementResultFloat64")
      .field("mesh", &arrangement_result_t<Real>::mesh)
      .field("tagLabels", &arrangement_result_t<Real>::tag_labels)
      .field("faceLabels", &arrangement_result_t<Real>::face_labels);

  emscripten::value_object<arrangement_result_with_curves_t<Real>>(
      "MeshArrangementResultWithCurvesFloat64")
      .field("mesh", &arrangement_result_with_curves_t<Real>::mesh)
      .field("tagLabels", &arrangement_result_with_curves_t<Real>::tag_labels)
      .field("faceLabels", &arrangement_result_with_curves_t<Real>::face_labels)
      .field("curves", &arrangement_result_with_curves_t<Real>::curves);

  // Sync
  emscripten::function("mesh_arrangements_float64",
                       &sync_mesh_arrangement<Real>);
  emscripten::function("mesh_arrangements_with_curves_float64",
                       &sync_mesh_arrangement_with_curves<Real>);

  // Async
  emscripten::function("dispatch_mesh_arrangements_float64",
                       &async_mesh_arrangement<Real>);
  emscripten::function("dispatch_mesh_arrangements_with_curves_float64",
                       &async_mesh_arrangement_with_curves<Real>);
}
