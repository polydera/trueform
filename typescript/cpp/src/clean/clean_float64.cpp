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

#include "./clean_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_clean_float64) {
  using Real = double;
  using namespace tf::ts;

  // Result types
  emscripten::value_object<cleaned_mesh_result_t<Real>>(
      "CleanedMeshResultFloat64")
      .field("mesh", &cleaned_mesh_result_t<Real>::mesh)
      .field("faceMap", &cleaned_mesh_result_t<Real>::face_map)
      .field("pointMap", &cleaned_mesh_result_t<Real>::point_map);

  emscripten::value_object<cleaned_points_result_t<Real>>(
      "CleanedPointsResultFloat64")
      .field("points", &cleaned_points_result_t<Real>::points)
      .field("pointMap", &cleaned_points_result_t<Real>::point_map);

  // Sync
  emscripten::function("cleaned_polygon_soup_float64",
                       &sync_cleaned_polygon_soup<Real>);
  emscripten::function("cleaned_mesh_float64", &sync_cleaned_mesh<Real>);
  emscripten::function("cleaned_mesh_with_maps_float64",
                       &sync_cleaned_mesh_with_maps<Real>);
  emscripten::function("cleaned_points_float64", &sync_cleaned_points<Real>);
  emscripten::function("cleaned_points_with_map_float64",
                       &sync_cleaned_points_with_map<Real>);

  // Async
  emscripten::function("dispatch_cleaned_polygon_soup_float64",
                       &async_cleaned_polygon_soup<Real>);
  emscripten::function("dispatch_cleaned_mesh_float64",
                       &async_cleaned_mesh<Real>);
  emscripten::function("dispatch_cleaned_mesh_with_maps_float64",
                       &async_cleaned_mesh_with_maps<Real>);
  emscripten::function("dispatch_cleaned_points_float64",
                       &async_cleaned_points<Real>);
  emscripten::function("dispatch_cleaned_points_with_map_float64",
                       &async_cleaned_points_with_map<Real>);
}
