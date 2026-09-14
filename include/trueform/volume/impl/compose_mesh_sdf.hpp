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
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace tf {
namespace volume_detail {

/// @brief Compose the banded field: the measured samples keep their measured
/// squared distances past the sweep, and every sample becomes its signed
/// distance, negative inside.
template <typename T, typename Vol>
void compose_mesh_sdf(Vol &vol, const tf::buffer<char> &inside,
                      tf::buffer<T> &field,
                      const tf::buffer<std::ptrdiff_t> &ids,
                      const tf::buffer<T> &measured) {
  tf::parallel_for_each(
      tf::make_sequence_range(std::ptrdiff_t(ids.size())),
      [&](std::ptrdiff_t bi) {
        field[std::size_t(ids[std::size_t(bi)])] = measured[std::size_t(bi)];
      });
  const auto nx = std::ptrdiff_t(vol.dims()[0]);
  const auto ny = std::ptrdiff_t(vol.dims()[1]);
  const auto rows =
      nx == 0 ? std::ptrdiff_t(0) : std::ptrdiff_t(field.size()) / nx;
  using sample_t = std::decay_t<decltype(vol(0, 0, 0))>;
  tf::parallel_for_each(
      tf::make_sequence_range(rows), [&](std::ptrdiff_t row) {
        const int y = int(row % ny), z = int(row / ny);
        const auto *in = inside.begin() + row * nx;
        const auto *f = field.begin() + row * nx;
        for (int x = 0; x < nx; ++x) {
          const double d = std::sqrt(double(f[x]));
          vol(x, y, z) = static_cast<sample_t>(in[x] ? -d : d);
        }
      });
}

} // namespace volume_detail
} // namespace tf
