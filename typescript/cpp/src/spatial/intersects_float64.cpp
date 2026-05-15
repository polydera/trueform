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

#include "./intersects_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_spatial_intersects_float64) {
  using Real = double;
  using namespace tf::ts;

  // PP
  emscripten::function("intersects_pp_float64", &sync_intersects_pp<Real>);
  emscripten::function("dispatch_intersects_pp_float64",
                       &async_intersects_pp<Real>);

  // FP — mesh & point cloud
  emscripten::function("intersects_fp_float64", &sync_intersects_fp<Real>);
  emscripten::function("intersects_fp_pc_float64",
                       &sync_intersects_fp_pc<Real>);
  emscripten::function("dispatch_intersects_fp_float64",
                       &async_intersects_fp<Real>);
  emscripten::function("dispatch_intersects_fp_pc_float64",
                       &async_intersects_fp_pc<Real>);

  // FF — all 4 combos
  emscripten::function("intersects_ff_float64", &sync_intersects_ff<Real>);
  emscripten::function("intersects_ff_mp_float64",
                       &sync_intersects_ff_mp<Real>);
  emscripten::function("intersects_ff_pm_float64",
                       &sync_intersects_ff_pm<Real>);
  emscripten::function("intersects_ff_pc_float64",
                       &sync_intersects_ff_pc<Real>);
  emscripten::function("dispatch_intersects_ff_float64",
                       &async_intersects_ff<Real>);
  emscripten::function("dispatch_intersects_ff_mp_float64",
                       &async_intersects_ff_mp<Real>);
  emscripten::function("dispatch_intersects_ff_pm_float64",
                       &async_intersects_ff_pm<Real>);
  emscripten::function("dispatch_intersects_ff_pc_float64",
                       &async_intersects_ff_pc<Real>);
}
