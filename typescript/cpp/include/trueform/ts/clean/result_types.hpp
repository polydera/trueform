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
#include "trueform/ts/core/wasm_mesh.hpp"

namespace tf {
namespace ts {

struct cleaned_mesh_result {
  wasm_mesh mesh;
  wasm_index_map face_map;
  wasm_index_map point_map;
};

struct cleaned_points_result {
  wasm_ndarray<float> points; // [N, 3]
  wasm_index_map point_map;
};

} // namespace ts
} // namespace tf
