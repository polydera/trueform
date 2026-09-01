/*
 * Copyright (c) 2026 XLAB
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

#include <emscripten/val.h>
#include <cstddef>
#include <vector>

namespace tf {
namespace ts {

/// emscripten::val cannot cross threads: extract on the main thread and
/// capture the resulting POD vector in async lambdas.
inline auto extract_int_vector(emscripten::val js_array) -> std::vector<int> {
  if (js_array.isUndefined() || js_array.isNull())
    return {};
  auto n = js_array["length"].as<int>();
  std::vector<int> out;
  out.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    out.push_back(js_array[i].as<int>());
  return out;
}

} // namespace ts
} // namespace tf
