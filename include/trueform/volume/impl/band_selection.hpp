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
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./dilate_mask.hpp"
#include "./grid_lines.hpp"
#include "./parity_inside.hpp"
#include <array>
#include <cstddef>

namespace tf {
namespace volume_detail {

/// @brief The samples the banded magnitude measures: the seeds dilated by
/// the band, the shell, and — when the shell says the surface leaves the
/// grid — every sample none of whose three lines an in-grid crossing
/// informed, since the discovery is blind there and only a measurement
/// answers. A surface the grid's lines resolve informs everything else
/// through its seeds' composite.
inline auto band_measured_mask(const band_discovery &discovery,
                               const std::array<bool, 3> &flips,
                               const std::array<std::ptrdiff_t, 3> &dims,
                               int band, bool leaves_grid)
    -> tf::buffer<char> {
  tf::buffer<char> measured;
  measured.allocate(discovery.seeds.size());
  tf::parallel_copy(discovery.seeds, measured);
  dilate_mask(measured, dims, band);
  const auto nx = dims[0], ny = dims[1], nz = dims[2];
  tf::parallel_for_each(
      tf::make_sequence_range(std::ptrdiff_t(measured.size())),
      [&](std::ptrdiff_t i) {
        char m =
            char(measured[std::size_t(i)] | discovery.shell[std::size_t(i)]);
        if (leaves_grid && !m) {
          const auto c = grid_coords(dims, i);
          const auto ax = flips[0] ? nx - 1 - c[0] : c[0];
          const auto ay = flips[1] ? ny - 1 - c[1] : c[1];
          const auto az = flips[2] ? nz - 1 - c[2] : c[2];
          const bool informed =
              discovery.rows_crossed[0][std::size_t(ay + ny * az)] ||
              discovery.rows_crossed[1][std::size_t(az + nz * ax)] ||
              discovery.rows_crossed[2][std::size_t(ax + nx * ay)];
          m = char(!informed);
        }
        measured[std::size_t(i)] = m;
      });
  return measured;
}

} // namespace volume_detail
} // namespace tf
