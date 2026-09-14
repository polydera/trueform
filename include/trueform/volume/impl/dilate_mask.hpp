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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./grid_lines.hpp"
#include <array>
#include <cstddef>
#include <utility>

namespace tf {
namespace volume_detail {

/// @brief Dilate a sample mask by `radius` voxels in the chebyshev metric:
/// one two-sweep pass per axis, each line marking everything within `radius`
/// of a set sample.
inline void dilate_mask(tf::buffer<char> &mask,
                        const std::array<std::ptrdiff_t, 3> &dims,
                        int radius) {
  if (radius <= 0)
    return;
  tf::buffer<char> scratch;
  scratch.allocate(mask.size());
  for (int axis = 0; axis < 3; ++axis) {
    const auto ext = dims[std::size_t(axis)];
    const auto lines =
        std::ptrdiff_t(mask.size()) / (ext == 0 ? std::ptrdiff_t(1) : ext);
    if (ext == 0 || lines == 0)
      continue;
    tf::parallel_for_each(
        tf::make_sequence_range(lines), [&, axis](std::ptrdiff_t line) {
          const auto [base, str] = grid_line_of(dims, axis, line);
          std::ptrdiff_t last = -(std::ptrdiff_t(radius) + 1);
          for (std::ptrdiff_t i = 0; i < ext; ++i) {
            if (mask[std::size_t(base + i * str)])
              last = i;
            scratch[std::size_t(base + i * str)] = char(i - last <= radius);
          }
          last = ext + radius + 1;
          for (std::ptrdiff_t i = ext - 1; i >= 0; --i) {
            if (mask[std::size_t(base + i * str)])
              last = i;
            if (last - i <= radius)
              scratch[std::size_t(base + i * str)] = 1;
          }
        });
    std::swap(mask, scratch);
  }
}

} // namespace volume_detail
} // namespace tf
