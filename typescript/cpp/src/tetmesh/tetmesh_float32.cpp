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

#include "./tetmesh_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_tetmesh_float32) {
  using Real = float;
  using namespace tf::ts;

  emscripten::function("tet_mesh_float32", &sync_tet_mesh<Real>);
  emscripten::function("dispatch_tet_mesh_float32", &async_tet_mesh<Real>);
  emscripten::function("tet_mesh_with_index_map_float32",
                       &sync_tet_mesh_with_index_map<Real>);
  emscripten::function("dispatch_tet_mesh_with_index_map_float32",
                       &async_tet_mesh_with_index_map<Real>);
}
