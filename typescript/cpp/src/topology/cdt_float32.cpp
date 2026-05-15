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

#include "./cdt_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_topology_cdt_float32) {
  using Real = float;
  using namespace tf::ts;

  // Result types
  emscripten::value_object<cdt_result_t<Real>>("CdtResultFloat32")
      .field("faces", &cdt_result_t<Real>::faces)
      .field("points", &cdt_result_t<Real>::points);

  emscripten::value_object<cdt_result_with_map_t<Real>>(
      "CdtResultWithMapFloat32")
      .field("faces", &cdt_result_with_map_t<Real>::faces)
      .field("points", &cdt_result_with_map_t<Real>::points)
      .field("indexMap", &cdt_result_with_map_t<Real>::index_map);

  // Sync
  emscripten::function("make_cdt_float32", &sync_make_cdt<Real>);
  emscripten::function("make_cdt_with_maps_float32",
                       &sync_make_cdt_with_maps<Real>);
  emscripten::function("make_cdt_edges_float32", &sync_make_cdt_edges<Real>);
  emscripten::function("make_cdt_edges_with_maps_float32",
                       &sync_make_cdt_edges_with_maps<Real>);
  emscripten::function("make_cdt_edges_masked_float32",
                       &sync_make_cdt_edges_masked<Real>);
  emscripten::function("make_cdt_edges_masked_with_maps_float32",
                       &sync_make_cdt_edges_masked_with_maps<Real>);

  // Async
  emscripten::function("dispatch_make_cdt_float32", &async_make_cdt<Real>);
  emscripten::function("dispatch_make_cdt_with_maps_float32",
                       &async_make_cdt_with_maps<Real>);
  emscripten::function("dispatch_make_cdt_edges_float32",
                       &async_make_cdt_edges<Real>);
  emscripten::function("dispatch_make_cdt_edges_with_maps_float32",
                       &async_make_cdt_edges_with_maps<Real>);
  emscripten::function("dispatch_make_cdt_edges_masked_float32",
                       &async_make_cdt_edges_masked<Real>);
  emscripten::function("dispatch_make_cdt_edges_masked_with_maps_float32",
                       &async_make_cdt_edges_masked_with_maps<Real>);
}
