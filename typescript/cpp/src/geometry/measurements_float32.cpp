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

#include "./measurements_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_geometry_measurements_float32) {
  using Real = float;
  using namespace tf::ts;

  // Sync
  emscripten::function("mesh_area_float32", &sync_area<Real>);
  emscripten::function("mesh_signed_volume_float32", &sync_signed_volume<Real>);
  emscripten::function("mesh_mean_edge_length_float32",
                       &sync_mean_edge_length<Real>);
  emscripten::function("mesh_min_edge_length_float32",
                       &sync_min_edge_length<Real>);
  emscripten::function("mesh_max_edge_length_float32",
                       &sync_max_edge_length<Real>);

  // Async
  emscripten::function("dispatch_mesh_area_float32", &async_area<Real>);
  emscripten::function("dispatch_mesh_signed_volume_float32",
                       &async_signed_volume<Real>);
  emscripten::function("dispatch_mesh_mean_edge_length_float32",
                       &async_mean_edge_length<Real>);
  emscripten::function("dispatch_mesh_min_edge_length_float32",
                       &async_min_edge_length<Real>);
  emscripten::function("dispatch_mesh_max_edge_length_float32",
                       &async_max_edge_length<Real>);
}
