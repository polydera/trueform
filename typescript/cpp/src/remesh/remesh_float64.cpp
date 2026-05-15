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

#include "./remesh_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_remesh_float64) {
  using Real = double;
  using namespace tf::ts;

  emscripten::function("decimated_float64", &sync_decimated<Real>);
  emscripten::function("isotropic_remeshed_float64",
                       &sync_isotropic_remeshed<Real>);

  emscripten::function("dispatch_decimated_float64", &async_decimated<Real>);
  emscripten::function("dispatch_isotropic_remeshed_float64",
                       &async_isotropic_remeshed<Real>);
}
