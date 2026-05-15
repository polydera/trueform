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

#include "./intersect_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_intersect_float32) {
  using Real = float;
  using namespace tf::ts;

  // Intersection curves — sync/async
  emscripten::function("intersection_curves_float32",
                       &sync_intersection_curves<Real>);
  emscripten::function("dispatch_intersection_curves_float32",
                       &async_intersection_curves<Real>);

  // Self-intersection curves — sync/async
  emscripten::function("self_intersection_curves_float32",
                       &sync_self_intersection_curves<Real>);
  emscripten::function("dispatch_self_intersection_curves_float32",
                       &async_self_intersection_curves<Real>);

  // Isocontours (single threshold) — sync/async
  emscripten::function("isocontours_float32", &sync_isocontours<Real>);
  emscripten::function("dispatch_isocontours_float32",
                       &async_isocontours<Real>);

  // Intersection curves list — sync/async
  emscripten::function("intersection_curves_list_float32",
                       &sync_intersection_curves_list<Real>);
  emscripten::function("dispatch_intersection_curves_list_float32",
                       &async_intersection_curves_list<Real>);

  // Isocontours (multiple thresholds) — sync/async
  emscripten::function("isocontours_multi_float32",
                       &sync_isocontours_multi<Real>);
  emscripten::function("dispatch_isocontours_multi_float32",
                       &async_isocontours_multi<Real>);
}
