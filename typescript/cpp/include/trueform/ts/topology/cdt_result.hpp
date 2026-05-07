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

#include "trueform/ts/core/wasm_index_map.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"

namespace tf {
namespace ts {

/// Output of tf::make_cdt: triangle list + 2D vertex coordinates.
struct cdt_result {
  wasm_ndarray<int> faces;     // [K, 3] int32 triangle indices
  wasm_ndarray<float> points;  // [M, 2] float32 vertex coordinates
};

/// Output of tf::make_cdt with `tf::return_index_map`: triangulation +
/// the input-point-to-output-point map.
struct cdt_result_with_map {
  wasm_ndarray<int> faces;
  wasm_ndarray<float> points;
  wasm_index_map index_map;
};

} // namespace ts
} // namespace tf
