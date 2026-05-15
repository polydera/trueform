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

#include "./smoothing_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_geometry_smoothing_float64) {
  using Real = double;
  using namespace tf::ts;

  emscripten::function("laplacian_smoothed_float64",
                       &sync_laplacian_smoothed<Real>);
  emscripten::function("dispatch_laplacian_smoothed_float64",
                       &async_laplacian_smoothed<Real>);
  emscripten::function("taubin_smoothed_float64",
                       &sync_taubin_smoothed<Real>);
  emscripten::function("dispatch_taubin_smoothed_float64",
                       &async_taubin_smoothed<Real>);
}
