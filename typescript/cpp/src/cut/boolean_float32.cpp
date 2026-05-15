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

#include "./boolean_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_cut_boolean_float32) {
  using Real = float;
  using namespace tf::ts;

  // Result types
  emscripten::value_object<labeled_cut_result_t<Real>>(
      "LabeledCutResultFloat32")
      .field("mesh", &labeled_cut_result_t<Real>::mesh)
      .field("labels", &labeled_cut_result_t<Real>::labels)
      .field("faceLabels", &labeled_cut_result_t<Real>::face_labels);

  emscripten::value_object<labeled_cut_result_with_curves_t<Real>>(
      "LabeledCutResultWithCurvesFloat32")
      .field("mesh", &labeled_cut_result_with_curves_t<Real>::mesh)
      .field("labels", &labeled_cut_result_with_curves_t<Real>::labels)
      .field("faceLabels", &labeled_cut_result_with_curves_t<Real>::face_labels)
      .field("curves", &labeled_cut_result_with_curves_t<Real>::curves);

  // Sync
  emscripten::function("boolean_union_float32", &sync_boolean_union<Real>);
  emscripten::function("boolean_intersection_float32",
                       &sync_boolean_intersection<Real>);
  emscripten::function("boolean_difference_float32",
                       &sync_boolean_difference<Real>);
  emscripten::function("boolean_union_with_curves_float32",
                       &sync_boolean_union_with_curves<Real>);
  emscripten::function("boolean_intersection_with_curves_float32",
                       &sync_boolean_intersection_with_curves<Real>);
  emscripten::function("boolean_difference_with_curves_float32",
                       &sync_boolean_difference_with_curves<Real>);

  // Async
  emscripten::function("dispatch_boolean_union_float32",
                       &async_boolean_union<Real>);
  emscripten::function("dispatch_boolean_intersection_float32",
                       &async_boolean_intersection<Real>);
  emscripten::function("dispatch_boolean_difference_float32",
                       &async_boolean_difference<Real>);
  emscripten::function("dispatch_boolean_union_with_curves_float32",
                       &async_boolean_union_with_curves<Real>);
  emscripten::function("dispatch_boolean_intersection_with_curves_float32",
                       &async_boolean_intersection_with_curves<Real>);
  emscripten::function("dispatch_boolean_difference_with_curves_float32",
                       &async_boolean_difference_with_curves<Real>);
}
