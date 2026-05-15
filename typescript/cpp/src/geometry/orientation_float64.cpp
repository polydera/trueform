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

#include "./orientation_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_geometry_orientation_float64) {
  using Real = double;
  using namespace tf::ts;

  emscripten::function("positively_oriented_float64",
                       &sync_positively_oriented<Real>);
  emscripten::function("dispatch_positively_oriented_float64",
                       &async_positively_oriented<Real>);
  emscripten::function("reverse_winding_float64", &sync_reverse_winding<Real>);
  emscripten::function("dispatch_reverse_winding_float64",
                       &async_reverse_winding<Real>);
}
