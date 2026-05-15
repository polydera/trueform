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

#include "./distance_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_distance_float32) {
  using Real = float;
  using namespace tf::ts;

  // PP
  emscripten::function("distance2_pp_float32", &sync_distance2_pp<Real>);
  emscripten::function("dispatch_distance2_pp_float32",
                       &async_distance2_pp<Real>);
  emscripten::function("distance_pp_float32", &sync_distance_pp<Real>);
  emscripten::function("dispatch_distance_pp_float32",
                       &async_distance_pp<Real>);

  // FP — mesh & point cloud
  emscripten::function("distance2_fp_float32", &sync_distance2_fp<Real>);
  emscripten::function("distance2_fp_pc_float32", &sync_distance2_fp_pc<Real>);
  emscripten::function("dispatch_distance2_fp_float32",
                       &async_distance2_fp<Real>);
  emscripten::function("dispatch_distance2_fp_pc_float32",
                       &async_distance2_fp_pc<Real>);

  emscripten::function("distance_fp_float32", &sync_distance_fp<Real>);
  emscripten::function("distance_fp_pc_float32", &sync_distance_fp_pc<Real>);
  emscripten::function("dispatch_distance_fp_float32",
                       &async_distance_fp<Real>);
  emscripten::function("dispatch_distance_fp_pc_float32",
                       &async_distance_fp_pc<Real>);

  // FF — all 4 combos
  emscripten::function("distance2_ff_float32", &sync_distance2_ff<Real>);
  emscripten::function("distance2_ff_mp_float32", &sync_distance2_ff_mp<Real>);
  emscripten::function("distance2_ff_pm_float32", &sync_distance2_ff_pm<Real>);
  emscripten::function("distance2_ff_pc_float32", &sync_distance2_ff_pc<Real>);
  emscripten::function("dispatch_distance2_ff_float32",
                       &async_distance2_ff<Real>);
  emscripten::function("dispatch_distance2_ff_mp_float32",
                       &async_distance2_ff_mp<Real>);
  emscripten::function("dispatch_distance2_ff_pm_float32",
                       &async_distance2_ff_pm<Real>);
  emscripten::function("dispatch_distance2_ff_pc_float32",
                       &async_distance2_ff_pc<Real>);

  emscripten::function("distance_ff_float32", &sync_distance_ff<Real>);
  emscripten::function("distance_ff_mp_float32", &sync_distance_ff_mp<Real>);
  emscripten::function("distance_ff_pm_float32", &sync_distance_ff_pm<Real>);
  emscripten::function("distance_ff_pc_float32", &sync_distance_ff_pc<Real>);
  emscripten::function("dispatch_distance_ff_float32",
                       &async_distance_ff<Real>);
  emscripten::function("dispatch_distance_ff_mp_float32",
                       &async_distance_ff_mp<Real>);
  emscripten::function("dispatch_distance_ff_pm_float32",
                       &async_distance_ff_pm<Real>);
  emscripten::function("dispatch_distance_ff_pc_float32",
                       &async_distance_ff_pc<Real>);
}
