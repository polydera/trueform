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
#include "trueform/ts/core/wasm_curves.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include <cstdint>

namespace tf {
namespace ts {

struct labeled_cut_result {
  wasm_mesh<float> mesh;
  wasm_ndarray<std::int8_t> labels;
  wasm_ndarray<std::int32_t> face_labels;
};

struct labeled_cut_result_with_curves {
  wasm_mesh<float> mesh;
  wasm_ndarray<std::int8_t> labels;
  wasm_ndarray<std::int32_t> face_labels;
  wasm_curves<float> curves;
};

struct cut_result {
  wasm_mesh<float> mesh;
  wasm_ndarray<std::int32_t> face_labels;
};

struct cut_result_with_curves {
  wasm_mesh<float> mesh;
  wasm_ndarray<std::int32_t> face_labels;
  wasm_curves<float> curves;
};

struct isobands_result {
  wasm_mesh<float> mesh;
  wasm_ndarray<int> labels;
  wasm_ndarray<std::int32_t> face_labels;
};

struct isobands_result_with_curves {
  wasm_mesh<float> mesh;
  wasm_ndarray<int> labels;
  wasm_ndarray<std::int32_t> face_labels;
  wasm_curves<float> curves;
};

} // namespace ts
} // namespace tf
