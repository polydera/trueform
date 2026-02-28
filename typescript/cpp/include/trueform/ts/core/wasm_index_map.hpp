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

#include "trueform/core/index_map.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"

namespace tf {
namespace ts {

struct wasm_index_map {
  wasm_ndarray<int> f;        // [original_count] old_id → new_id
  wasm_ndarray<int> kept_ids; // [kept_count] original IDs that survived

  /// Build from a C++ index_map_buffer<int>, consuming its buffers.
  static auto from_index_map_buffer(tf::index_map_buffer<int> &&im)
      -> wasm_index_map {
    auto f_size = static_cast<int>(im.f().size());
    auto k_size = static_cast<int>(im.kept_ids().size());
    return {
        wasm_ndarray<int>::from_buffer(std::move(im.f()), {f_size}),
        wasm_ndarray<int>::from_buffer(std::move(im.kept_ids()), {k_size}),
    };
  }
};

} // namespace ts
} // namespace tf
