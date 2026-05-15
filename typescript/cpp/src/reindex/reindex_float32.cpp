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

#include "./reindex_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_reindex_float32) {
  using Real = float;
  using namespace tf::ts;

  // Result types
  emscripten::value_object<reindexed_mesh_result_t<Real>>(
      "ReindexedMeshResultFloat32")
      .field("mesh", &reindexed_mesh_result_t<Real>::mesh)
      .field("faceMap", &reindexed_mesh_result_t<Real>::face_map)
      .field("pointMap", &reindexed_mesh_result_t<Real>::point_map);

  emscripten::value_object<split_components_result_t<Real>>(
      "SplitResultFloat32")
      .field("components", &split_components_result_t<Real>::components)
      .field("labels", &split_components_result_t<Real>::labels);

  // reindexed(mesh, faceMap, pointMap)
  emscripten::function("reindexed_mesh_float32", &sync_reindexed_mesh<Real>);
  emscripten::function("dispatch_reindexed_mesh_float32",
                       &async_reindexed_mesh<Real>);

  // reindexed_by_mask
  emscripten::function("reindexed_by_mask_float32",
                       &sync_reindexed_by_mask<Real>);
  emscripten::function("reindexed_by_mask_with_maps_float32",
                       &sync_reindexed_by_mask_with_maps<Real>);
  emscripten::function("dispatch_reindexed_by_mask_float32",
                       &async_reindexed_by_mask<Real>);
  emscripten::function("dispatch_reindexed_by_mask_with_maps_float32",
                       &async_reindexed_by_mask_with_maps<Real>);

  // reindexed_by_ids
  emscripten::function("reindexed_by_ids_float32",
                       &sync_reindexed_by_ids<Real>);
  emscripten::function("reindexed_by_ids_with_maps_float32",
                       &sync_reindexed_by_ids_with_maps<Real>);
  emscripten::function("dispatch_reindexed_by_ids_float32",
                       &async_reindexed_by_ids<Real>);
  emscripten::function("dispatch_reindexed_by_ids_with_maps_float32",
                       &async_reindexed_by_ids_with_maps<Real>);

  // reindexed_by_mask_on_points
  emscripten::function("reindexed_by_mask_on_points_float32",
                       &sync_reindexed_by_mask_on_points<Real>);
  emscripten::function("reindexed_by_mask_on_points_with_maps_float32",
                       &sync_reindexed_by_mask_on_points_with_maps<Real>);
  emscripten::function("dispatch_reindexed_by_mask_on_points_float32",
                       &async_reindexed_by_mask_on_points<Real>);
  emscripten::function("dispatch_reindexed_by_mask_on_points_with_maps_float32",
                       &async_reindexed_by_mask_on_points_with_maps<Real>);

  // reindexed_by_ids_on_points
  emscripten::function("reindexed_by_ids_on_points_float32",
                       &sync_reindexed_by_ids_on_points<Real>);
  emscripten::function("reindexed_by_ids_on_points_with_maps_float32",
                       &sync_reindexed_by_ids_on_points_with_maps<Real>);
  emscripten::function("dispatch_reindexed_by_ids_on_points_float32",
                       &async_reindexed_by_ids_on_points<Real>);
  emscripten::function("dispatch_reindexed_by_ids_on_points_with_maps_float32",
                       &async_reindexed_by_ids_on_points_with_maps<Real>);

  // concatenate_meshes
  emscripten::function("concatenate_meshes_float32",
                       &sync_concatenate_meshes<Real>);
  emscripten::function("dispatch_concatenate_meshes_float32",
                       &async_concatenate_meshes<Real>);

  // split_into_components
  emscripten::function("split_into_components_float32",
                       &sync_split_into_components<Real>);
  emscripten::function("dispatch_split_into_components_float32",
                       &async_split_into_components<Real>);
}
