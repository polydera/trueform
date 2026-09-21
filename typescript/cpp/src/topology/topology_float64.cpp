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

EMSCRIPTEN_BINDINGS(trueform_topology_float64) {
  using Real = double;
  using namespace tf::ts;

  emscripten::value_object<boundary_rims_result_t<Real>>(
      "BoundaryRimsResultFloat64")
      .field("vertices", &boundary_rims_result_t<Real>::vertices)
      .field("faces", &boundary_rims_result_t<Real>::faces)
      .field("closed", &boundary_rims_result_t<Real>::closed);

  emscripten::value_object<split_non_manifold_vertices_result_t<Real>>(
      "SplitNonManifoldVerticesResultFloat64")
      .field("mesh", &split_non_manifold_vertices_result_t<Real>::mesh)
      .field("pointMap",
             &split_non_manifold_vertices_result_t<Real>::point_map);

  // Boolean queries
  emscripten::function("is_closed_float64", &sync_is_closed<Real>);
  emscripten::function("dispatch_is_closed_float64", &async_is_closed<Real>);
  emscripten::function("is_open_float64", &sync_is_open<Real>);
  emscripten::function("dispatch_is_open_float64", &async_is_open<Real>);
  emscripten::function("is_manifold_float64", &sync_is_manifold<Real>);
  emscripten::function("dispatch_is_manifold_float64",
                       &async_is_manifold<Real>);
  emscripten::function("is_non_manifold_float64", &sync_is_non_manifold<Real>);
  emscripten::function("dispatch_is_non_manifold_float64",
                       &async_is_non_manifold<Real>);

  // Scalar
  emscripten::function("euler_characteristic_float64",
                       &sync_euler_characteristic<Real>);
  emscripten::function("dispatch_euler_characteristic_float64",
                       &async_euler_characteristic<Real>);

  // Edge arrays
  emscripten::function("boundary_edges_float64", &sync_boundary_edges<Real>);
  emscripten::function("dispatch_boundary_edges_float64",
                       &async_boundary_edges<Real>);
  emscripten::function("non_manifold_edges_float64",
                       &sync_non_manifold_edges<Real>);
  emscripten::function("dispatch_non_manifold_edges_float64",
                       &async_non_manifold_edges<Real>);

  // Vertex arrays
  emscripten::function("non_manifold_vertices_float64",
                       &sync_non_manifold_vertices<Real>);
  emscripten::function("dispatch_non_manifold_vertices_float64",
                       &async_non_manifold_vertices<Real>);

  // OffsetBlockedBuffer results
  emscripten::function("boundary_paths_float64", &sync_boundary_paths<Real>);
  emscripten::function("dispatch_boundary_paths_float64",
                       &async_boundary_paths<Real>);
  emscripten::function("boundary_rims_float64", &sync_boundary_rims<Real>);
  emscripten::function("dispatch_boundary_rims_float64",
                       &async_boundary_rims<Real>);
  emscripten::function("k_rings_float64", &sync_k_rings<Real>);
  emscripten::function("dispatch_k_rings_float64", &async_k_rings<Real>);
  emscripten::function("neighborhoods_float64", &sync_neighborhoods<Real>);
  emscripten::function("dispatch_neighborhoods_float64",
                       &async_neighborhoods<Real>);

  // Mesh mutation
  emscripten::function("consistently_oriented_float64",
                       &sync_consistently_oriented<Real>);
  emscripten::function("dispatch_consistently_oriented_float64",
                       &async_consistently_oriented<Real>);
  emscripten::function("split_non_manifold_vertices_float64",
                       &sync_split_non_manifold_vertices<Real>);
  emscripten::function("dispatch_split_non_manifold_vertices_float64",
                       &async_split_non_manifold_vertices<Real>);
}
