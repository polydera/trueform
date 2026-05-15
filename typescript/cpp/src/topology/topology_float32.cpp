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

#include "./topology_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_topology_float32) {
  using Real = float;
  using namespace tf::ts;

  emscripten::value_object<connected_components_result>(
      "ConnectedComponentsResult")
      .field("labels", &connected_components_result::labels)
      .field("nComponents", &connected_components_result::n_components);

  // Boolean queries
  emscripten::function("is_closed_float32", &sync_is_closed<Real>);
  emscripten::function("dispatch_is_closed_float32", &async_is_closed<Real>);
  emscripten::function("is_open_float32", &sync_is_open<Real>);
  emscripten::function("dispatch_is_open_float32", &async_is_open<Real>);
  emscripten::function("is_manifold_float32", &sync_is_manifold<Real>);
  emscripten::function("dispatch_is_manifold_float32",
                       &async_is_manifold<Real>);
  emscripten::function("is_non_manifold_float32", &sync_is_non_manifold<Real>);
  emscripten::function("dispatch_is_non_manifold_float32",
                       &async_is_non_manifold<Real>);

  // Scalar
  emscripten::function("euler_characteristic_float32",
                       &sync_euler_characteristic<Real>);
  emscripten::function("dispatch_euler_characteristic_float32",
                       &async_euler_characteristic<Real>);

  // Edge arrays
  emscripten::function("boundary_edges_float32", &sync_boundary_edges<Real>);
  emscripten::function("dispatch_boundary_edges_float32",
                       &async_boundary_edges<Real>);
  emscripten::function("non_manifold_edges_float32",
                       &sync_non_manifold_edges<Real>);
  emscripten::function("dispatch_non_manifold_edges_float32",
                       &async_non_manifold_edges<Real>);

  // OffsetBlockedBuffer results
  emscripten::function("boundary_paths_float32", &sync_boundary_paths<Real>);
  emscripten::function("dispatch_boundary_paths_float32",
                       &async_boundary_paths<Real>);
  emscripten::function("k_rings_float32", &sync_k_rings<Real>);
  emscripten::function("dispatch_k_rings_float32", &async_k_rings<Real>);
  emscripten::function("neighborhoods_float32", &sync_neighborhoods<Real>);
  emscripten::function("dispatch_neighborhoods_float32",
                       &async_neighborhoods<Real>);

  // Non-templated (int32 inputs only) — registered once.
  emscripten::function("connect_edges_to_paths", &sync_connect_edges_to_paths);
  emscripten::function("dispatch_connect_edges_to_paths",
                       &async_connect_edges_to_paths);
  emscripten::function("label_connected_components_obb",
                       &sync_label_connected_components_obb);
  emscripten::function("dispatch_label_connected_components_obb",
                       &async_label_connected_components_obb);
  emscripten::function("label_connected_components_ndarray",
                       &sync_label_connected_components_ndarray);
  emscripten::function("dispatch_label_connected_components_ndarray",
                       &async_label_connected_components_ndarray);

  // Mesh mutation
  emscripten::function("consistently_oriented_float32",
                       &sync_consistently_oriented<Real>);
  emscripten::function("dispatch_consistently_oriented_float32",
                       &async_consistently_oriented<Real>);
}
