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
#pragma once

#include "trueform/ts/core/wasm_ndarray.hpp"
#include <cstdint>

namespace tf {
namespace ts {

// ============================================================================
// Helper: copy a tf::point<float,3> into a fresh wasm_ndarray<float> [3]
// ============================================================================

template <typename PointLike>
inline auto point_to_ndarray(const PointLike &pt) -> wasm_ndarray<float> {
  tf::buffer<float> buf;
  buf.allocate(3);
  auto *p = buf.data();
  p[0] = pt[0];
  p[1] = pt[1];
  p[2] = pt[2];
  return wasm_ndarray<float>::from_buffer(std::move(buf), {3});
}

// ============================================================================
// Result types for spatial queries
// ============================================================================

struct metric_point_result {
  wasm_ndarray<float> point; // [3]
  float distance2;
};

struct metric_point_pair_result {
  wasm_ndarray<float> point0; // [3]
  wasm_ndarray<float> point1; // [3]
  float distance2;
};

struct metric_point_batch_result {
  wasm_ndarray<float> points;    // [N, 3]
  wasm_ndarray<float> distances; // [N]
};

struct metric_point_pair_batch_result {
  wasm_ndarray<float> points0;   // [N, 3]
  wasm_ndarray<float> points1;   // [N, 3]
  wasm_ndarray<float> distances; // [N]
};

struct ray_cast_result {
  bool hit;
  float t;
  int element_id; // -1 for primitive targets
};

struct ray_cast_prim_batch_result {
  wasm_ndarray<std::int8_t> hits; // [N]
  wasm_ndarray<float> ts;        // [N]
};

struct ray_cast_form_batch_result {
  wasm_ndarray<std::int8_t> hits; // [N]
  wasm_ndarray<float> ts;        // [N]
  wasm_ndarray<int> element_ids;  // [N]
};

struct neighbor_result {
  int element_id; // -1 if no hit
  float distance2;
  wasm_ndarray<float> point; // [3], closest point on form
};

struct neighbor_batch_result {
  wasm_ndarray<int> element_ids;   // [N]
  wasm_ndarray<float> points;     // [N, 3]
  wasm_ndarray<float> distances;  // [N]
};

struct neighbor_result_pair {
  int element_id0; // -1 if no hit
  int element_id1;
  float distance2;
  wasm_ndarray<float> point0; // [3], closest point on form0
  wasm_ndarray<float> point1; // [3], closest point on form1
};

struct neighbor_knn_result {
  wasm_ndarray<int> element_ids; // [count]
  wasm_ndarray<float> points;   // [count, 3]
  wasm_ndarray<float> distances; // [count]
};

struct neighbor_knn_batch_result {
  wasm_ndarray<int> element_ids; // [N, k] padded with -1
  wasm_ndarray<float> points;   // [N, k, 3]
  wasm_ndarray<float> distances; // [N, k]
  wasm_ndarray<int> counts;     // [N] actual count per query
};

} // namespace ts
} // namespace tf
