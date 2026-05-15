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

#include "./registration_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_geometry_registration_float32) {
  using Real = float;
  using namespace tf::ts;

  emscripten::function("fit_icp_alignment_float32", &sync_fit_icp<Real>);
  emscripten::function("fit_rigid_alignment_float32", &sync_fit_rigid<Real>);
  emscripten::function("fit_obb_alignment_float32", &sync_fit_obb<Real>);
  emscripten::function("chamfer_error_float32", &sync_chamfer_error<Real>);
  emscripten::function("dispatch_fit_icp_alignment_float32",
                       &async_fit_icp<Real>);
  emscripten::function("dispatch_fit_rigid_alignment_float32",
                       &async_fit_rigid<Real>);
  emscripten::function("dispatch_fit_obb_alignment_float32",
                       &async_fit_obb<Real>);
  emscripten::function("dispatch_chamfer_error_float32",
                       &async_chamfer_error<Real>);
}
