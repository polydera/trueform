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
#include <array>
#include <cstddef>

namespace tf {
namespace volume_detail {

/// @brief How many samples a grid of @p dims holds.
///
/// The one producer of that count: the view's length, the buffer's allocation
/// and every boundary's validation read it, so an axis with no extent states
/// an empty grid once instead of wrapping a product around.
template <std::size_t Dims>
inline auto grid_sample_count(const std::array<int, Dims> &dims)
    -> std::size_t {
  std::size_t count = 1;
  for (std::size_t i = 0; i < Dims; ++i) {
    if (dims[i] <= 0)
      return 0;
    count *= static_cast<std::size_t>(dims[i]);
  }
  return count;
}

} // namespace volume_detail
} // namespace tf
