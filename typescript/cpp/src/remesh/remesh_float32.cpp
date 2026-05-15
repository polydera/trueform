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

EMSCRIPTEN_BINDINGS(trueform_remesh_float32) {
  using Real = float;
  using namespace tf::ts;

  emscripten::function("decimated_float32", &sync_decimated<Real>);
  emscripten::function("isotropic_remeshed_float32",
                       &sync_isotropic_remeshed<Real>);

  emscripten::function("dispatch_decimated_float32", &async_decimated<Real>);
  emscripten::function("dispatch_isotropic_remeshed_float32",
                       &async_isotropic_remeshed<Real>);
}
