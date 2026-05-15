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

#include "./curvature_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_geometry_curvature_float64) {
  using Real = double;
  using namespace tf::ts;

  emscripten::value_object<curvature_result_t<Real>>("CurvatureResultFloat64")
      .field("k0", &curvature_result_t<Real>::k0)
      .field("k1", &curvature_result_t<Real>::k1);

  emscripten::value_object<curvature_directions_result_t<Real>>(
      "CurvatureDirectionsResultFloat64")
      .field("k0", &curvature_directions_result_t<Real>::k0)
      .field("k1", &curvature_directions_result_t<Real>::k1)
      .field("d0", &curvature_directions_result_t<Real>::d0)
      .field("d1", &curvature_directions_result_t<Real>::d1);

  // Sync
  emscripten::function("principal_curvatures_float64",
                       &sync_principal_curvatures<Real>);
  emscripten::function("principal_directions_float64",
                       &sync_principal_directions<Real>);
  emscripten::function("shape_index_float64", &sync_shape_index<Real>);

  // Async
  emscripten::function("dispatch_principal_curvatures_float64",
                       &async_principal_curvatures<Real>);
  emscripten::function("dispatch_principal_directions_float64",
                       &async_principal_directions<Real>);
  emscripten::function("dispatch_shape_index_float64",
                       &async_shape_index<Real>);
}
