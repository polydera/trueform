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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./grid_lines.hpp"
#include "./mask_ids.hpp"
#include "./parity_inside.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>

namespace tf {
namespace volume_detail {

/// @brief The samples the shell's off-grid statements own: walking inward
/// from a shell sample, while its measured distance plus the steps
/// under-runs the swept field, the sweep cannot know the nearer surface
/// past the grid and only a measurement answers. Returns their ids;
/// samples already measured are passed through, not claimed.
template <typename T>
auto shell_walk(const band_discovery &discovery,
                const tf::buffer<char> &measured_mask,
                const tf::buffer<std::ptrdiff_t> &ids,
                const tf::buffer<T> &measured, const tf::buffer<T> &field,
                const std::array<std::ptrdiff_t, 3> &dims,
                const std::array<double, 3> &steps)
    -> tf::buffer<std::ptrdiff_t> {
  const auto shell_ids = mask_ids(discovery.shell);
  tf::buffer<char> owned;
  owned.allocate(field.size());
  tf::parallel_fill(owned, char(0));
  const auto strides = grid_strides(dims);
  tf::parallel_for_each(
      tf::make_sequence_range(std::ptrdiff_t(shell_ids.size())),
      [&](std::ptrdiff_t si) {
        const auto i = shell_ids[std::size_t(si)];
        const auto bi = std::lower_bound(ids.begin(), ids.end(), i);
        // the shell is part of the measured set, so the search must hit
        assert(bi != ids.end() && *bi == i);
        const double f = std::sqrt(double(measured[std::size_t(bi - ids.begin())]));
        const auto c = grid_coords(dims, i);
        for (int axis = 0; axis < 3; ++axis)
          for (int dir = -1; dir <= 1; dir += 2) {
            for (std::ptrdiff_t k = 1;; ++k) {
              const auto coord = c[std::size_t(axis)] + dir * k;
              if (coord < 0 || coord >= dims[std::size_t(axis)])
                break;
              const auto j = i + dir * k * strides[std::size_t(axis)];
              const double bound = f + double(k) * steps[std::size_t(axis)];
              if (T(bound * bound) >= field[std::size_t(j)])
                break;
              if (!measured_mask[std::size_t(j)])
                owned[std::size_t(j)] = 1;
            }
          }
      });
  return mask_ids(owned);
}

} // namespace volume_detail
} // namespace tf
