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

#include "./mesh_primitives_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_geometry_mesh_primitives_float32) {
  using Real = float;
  using namespace tf::ts;

  // Sync
  emscripten::function("make_sphere_mesh_float32",
                       &sync_make_sphere_mesh<Real>);
  emscripten::function("make_cylinder_mesh_float32",
                       &sync_make_cylinder_mesh<Real>);
  emscripten::function("make_box_mesh_float32", &sync_make_box_mesh<Real>);
  emscripten::function("make_box_mesh_subdivided_float32",
                       &sync_make_box_mesh_subdivided<Real>);
  emscripten::function("make_plane_mesh_float32", &sync_make_plane_mesh<Real>);
  emscripten::function("make_tube_mesh_float32", &sync_make_tube_mesh<Real>);

  // Async
  emscripten::function("dispatch_make_sphere_mesh_float32",
                       &async_make_sphere_mesh<Real>);
  emscripten::function("dispatch_make_cylinder_mesh_float32",
                       &async_make_cylinder_mesh<Real>);
  emscripten::function("dispatch_make_box_mesh_float32",
                       &async_make_box_mesh<Real>);
  emscripten::function("dispatch_make_box_mesh_subdivided_float32",
                       &async_make_box_mesh_subdivided<Real>);
  emscripten::function("dispatch_make_plane_mesh_float32",
                       &async_make_plane_mesh<Real>);
  emscripten::function("dispatch_make_tube_mesh_float32",
                       &async_make_tube_mesh<Real>);
}
