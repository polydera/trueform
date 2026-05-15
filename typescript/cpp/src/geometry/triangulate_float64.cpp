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

#include "./triangulate_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_geometry_triangulate_float64) {
  using Real = double;
  using namespace tf::ts;

  // Sync
  emscripten::function("triangulate_mesh_float64", &sync_triangulate_mesh<Real>);
  emscripten::function("triangulate_dynamic_float64",
                       &sync_triangulate_dynamic<Real>);
  emscripten::function("triangulate_fixed_float64",
                       &sync_triangulate_fixed<Real>);
  emscripten::function("triangulate_polygon_float64",
                       &sync_triangulate_polygon<Real>);

  // Async
  emscripten::function("dispatch_triangulate_mesh_float64",
                       &async_triangulate_mesh<Real>);
  emscripten::function("dispatch_triangulate_dynamic_float64",
                       &async_triangulate_dynamic<Real>);
  emscripten::function("dispatch_triangulate_fixed_float64",
                       &async_triangulate_fixed<Real>);
  emscripten::function("dispatch_triangulate_polygon_float64",
                       &async_triangulate_polygon<Real>);
}
