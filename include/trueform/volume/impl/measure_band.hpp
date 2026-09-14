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
#include "../../spatial/neighbor_search.hpp"
#include "./grid_lines.hpp"
#include <array>
#include <cmath>
#include <cstddef>

namespace tf {
namespace volume_detail {

/// @brief Measure the listed samples against the surface: squared distances
/// into `measured`, aligned with `ids`. Within a row the field is
/// 1-Lipschitz per step, so the previous sample's answer bounds the next
/// search; a bounded search that still misses falls back to the unbounded
/// one.
template <typename T, typename Form, typename Vol>
void measure_band(const Form &dform, const Vol &vol,
                  const tf::buffer<std::ptrdiff_t> &ids, double step_x,
                  tf::buffer<T> &measured) {
  measured.allocate(ids.size());
  const std::array<std::ptrdiff_t, 3> dims{std::ptrdiff_t(vol.dims()[0]),
                                           std::ptrdiff_t(vol.dims()[1]),
                                           std::ptrdiff_t(vol.dims()[2])};
  const auto nx = dims[0];
  struct warm_t {
    std::ptrdiff_t last_i = -2;
    double last_d = 0;
  };
  tf::parallel_for_each(
      tf::make_sequence_range(std::ptrdiff_t(ids.size())),
      [&](std::ptrdiff_t bi, warm_t &warm) {
        const auto i = ids[std::size_t(bi)];
        const auto c = grid_coords(dims, i);
        const auto p =
            vol.template point_at<double>(int(c[0]), int(c[1]), int(c[2]));
        double d = 0;
        bool measured_one = false;
        if (warm.last_i + 1 == i && warm.last_i / nx == i / nx) {
          const auto proj =
              tf::neighbor_search(dform, p, (warm.last_d + step_x) * 1.0001);
          if (proj) {
            d = std::sqrt(proj.metric());
            measured_one = true;
          }
        }
        if (!measured_one) {
          const auto proj = tf::neighbor_search(dform, p);
          d = proj ? std::sqrt(proj.metric()) : 0.0;
        }
        warm.last_i = i;
        warm.last_d = d;
        measured[std::size_t(bi)] = T(d * d);
      },
      warm_t{});
}

} // namespace volume_detail
} // namespace tf
