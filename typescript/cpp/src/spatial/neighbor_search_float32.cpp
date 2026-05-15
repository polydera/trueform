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

#include "./neighbor_search_impl.hpp"
#include <emscripten/bind.h>

EMSCRIPTEN_BINDINGS(trueform_spatial_neighbor_search_float32) {
  using Real = float;
  using namespace tf::ts;

  emscripten::value_object<neighbor_result_t<Real>>("NeighborResultFloat32")
      .field("elementId", &neighbor_result_t<Real>::element_id)
      .field("distance2", &neighbor_result_t<Real>::distance2)
      .field("point", &neighbor_result_t<Real>::point);

  emscripten::value_object<neighbor_batch_result_t<Real>>(
      "NeighborBatchResultFloat32")
      .field("elementIds", &neighbor_batch_result_t<Real>::element_ids)
      .field("points", &neighbor_batch_result_t<Real>::points)
      .field("distances", &neighbor_batch_result_t<Real>::distances);

  emscripten::value_object<neighbor_result_pair_t<Real>>(
      "NeighborResultPairFloat32")
      .field("elementId0", &neighbor_result_pair_t<Real>::element_id0)
      .field("elementId1", &neighbor_result_pair_t<Real>::element_id1)
      .field("distance2", &neighbor_result_pair_t<Real>::distance2)
      .field("point0", &neighbor_result_pair_t<Real>::point0)
      .field("point1", &neighbor_result_pair_t<Real>::point1);

  emscripten::value_object<neighbor_knn_result_t<Real>>(
      "NeighborKnnResultFloat32")
      .field("elementIds", &neighbor_knn_result_t<Real>::element_ids)
      .field("points", &neighbor_knn_result_t<Real>::points)
      .field("distances", &neighbor_knn_result_t<Real>::distances);

  emscripten::value_object<neighbor_knn_batch_result_t<Real>>(
      "NeighborKnnBatchResultFloat32")
      .field("elementIds", &neighbor_knn_batch_result_t<Real>::element_ids)
      .field("points", &neighbor_knn_batch_result_t<Real>::points)
      .field("distances", &neighbor_knn_batch_result_t<Real>::distances)
      .field("counts", &neighbor_knn_batch_result_t<Real>::counts);

  // FP — mesh & point cloud
  emscripten::function("neighbor_search_fp_float32",
                       &sync_neighbor_search_fp<Real>);
  emscripten::function("neighbor_search_fp_pc_float32",
                       &sync_neighbor_search_fp_pc<Real>);
  emscripten::function("dispatch_neighbor_search_fp_float32",
                       &async_neighbor_search_fp<Real>);
  emscripten::function("dispatch_neighbor_search_fp_pc_float32",
                       &async_neighbor_search_fp_pc<Real>);

  // FP k-NN — mesh & point cloud
  emscripten::function("neighbor_search_fp_knn_float32",
                       &sync_neighbor_search_fp_knn<Real>);
  emscripten::function("neighbor_search_fp_knn_pc_float32",
                       &sync_neighbor_search_fp_knn_pc<Real>);
  emscripten::function("dispatch_neighbor_search_fp_knn_float32",
                       &async_neighbor_search_fp_knn<Real>);
  emscripten::function("dispatch_neighbor_search_fp_knn_pc_float32",
                       &async_neighbor_search_fp_knn_pc<Real>);

  // FF — all 4 combos
  emscripten::function("neighbor_search_ff_float32",
                       &sync_neighbor_search_ff<Real>);
  emscripten::function("neighbor_search_ff_mp_float32",
                       &sync_neighbor_search_ff_mp<Real>);
  emscripten::function("neighbor_search_ff_pm_float32",
                       &sync_neighbor_search_ff_pm<Real>);
  emscripten::function("neighbor_search_ff_pc_float32",
                       &sync_neighbor_search_ff_pc<Real>);
  emscripten::function("dispatch_neighbor_search_ff_float32",
                       &async_neighbor_search_ff<Real>);
  emscripten::function("dispatch_neighbor_search_ff_mp_float32",
                       &async_neighbor_search_ff_mp<Real>);
  emscripten::function("dispatch_neighbor_search_ff_pm_float32",
                       &async_neighbor_search_ff_pm<Real>);
  emscripten::function("dispatch_neighbor_search_ff_pc_float32",
                       &async_neighbor_search_ff_pc<Real>);
}
