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

#include "./edges_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_geometry_edges_float64) {
  using Real = double;
  using namespace tf::ts;

  emscripten::function("sharp_edges_float64", &sync_sharp_edges<Real>);
  emscripten::function("dispatch_sharp_edges_float64", &async_sharp_edges<Real>);
}
