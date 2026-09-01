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

#include "./isocontours_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_iso_isocontours_float64) {
  using Real = double;
  using namespace tf::ts;

  // Isocontours (single threshold) — sync/async
  emscripten::function("isocontours_float64", &sync_isocontours<Real>);
  emscripten::function("dispatch_isocontours_float64",
                       &async_isocontours<Real>);

  // Isocontours (multiple thresholds) — sync/async
  emscripten::function("isocontours_multi_float64",
                       &sync_isocontours_multi<Real>);
  emscripten::function("dispatch_isocontours_multi_float64",
                       &async_isocontours_multi<Real>);
}
