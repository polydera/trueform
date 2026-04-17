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

namespace tf {
namespace ts {

/// @brief Histogram result with int32 counts (unweighted, non-density).
struct histogram_result_int {
  wasm_ndarray<int> counts;
  wasm_ndarray<float> edges;
};

/// @brief Histogram result with float32 counts (weighted OR density).
struct histogram_result_float {
  wasm_ndarray<float> counts;
  wasm_ndarray<float> edges;
};

} // namespace ts
} // namespace tf
