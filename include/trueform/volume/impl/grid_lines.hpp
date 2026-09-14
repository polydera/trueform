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

/// The one producer of a grid's line addressing: line `n` of `axis` starts
/// at `base` and steps by `step` through the x-fastest sample order the
/// volume carrier states.
struct grid_line {
  std::ptrdiff_t base;
  std::ptrdiff_t step;
};

inline auto grid_strides(const std::array<std::ptrdiff_t, 3> &dims)
    -> std::array<std::ptrdiff_t, 3> {
  return {1, dims[0], dims[0] * dims[1]};
}

/// The flat sample id decomposed into its grid coordinates.
inline auto grid_coords(const std::array<std::ptrdiff_t, 3> &dims,
                        std::ptrdiff_t i) -> std::array<std::ptrdiff_t, 3> {
  return {i % dims[0], (i / dims[0]) % dims[1], i / (dims[0] * dims[1])};
}

inline auto grid_line_of(const std::array<std::ptrdiff_t, 3> &dims, int axis,
                         std::ptrdiff_t line) -> grid_line {
  const auto stride = grid_strides(dims);
  std::ptrdiff_t base;
  if (axis == 0)
    base = line * dims[0];
  else if (axis == 1)
    base = line % dims[0] + dims[0] * dims[1] * (line / dims[0]);
  else
    base = line;
  return {base, stride[std::size_t(axis)]};
}

} // namespace volume_detail
} // namespace tf
