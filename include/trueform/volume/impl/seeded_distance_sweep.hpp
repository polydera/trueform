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
#include <cmath>
#include <cstddef>
#include <limits>

namespace tf {
namespace volume_detail {

/// @brief The seeded squared-distance sweep: Felzenszwalb-Huttenlocher lower
/// envelopes, one separable pass per axis of a three-axis grid.
///
/// `field` holds squared distances at seed samples and infinity elsewhere;
/// after the three passes every sample holds `min` over seeds of the seed's
/// value plus the anisotropic squared lattice distance, `weights[d]` being
/// the squared physical step of axis `d`. A line without a seed is left as
/// it is.
template <typename T>
void seeded_distance_sweep_axis(tf::buffer<T> &field,
                                const std::array<std::ptrdiff_t, 3> &dims,
                                int axis, double weight) {
  const auto ext = dims[std::size_t(axis)];
  const auto lines =
      std::ptrdiff_t(field.size()) / (ext == 0 ? std::ptrdiff_t(1) : ext);
  if (ext == 0 || lines == 0)
    return;
  struct local_t {
    tf::buffer<T> f;
    tf::buffer<int> sites;
    tf::buffer<double> bounds;
  };
  tf::parallel_for_each(
      tf::make_sequence_range(lines),
      [&, axis](std::ptrdiff_t line, local_t &local) {
        const auto [base, str] = grid_line_of(dims, axis, line);
        local.f.allocate(std::size_t(ext));
        local.sites.allocate(std::size_t(ext));
        local.bounds.allocate(std::size_t(ext) + 1);
        for (std::ptrdiff_t i = 0; i < ext; ++i)
          local.f[std::size_t(i)] = field[std::size_t(base + i * str)];
        int m = 0;
        auto sect = [&](int a, int b) {
          return (double(local.f[std::size_t(a)]) + weight * double(a) * a -
                  double(local.f[std::size_t(b)]) - weight * double(b) * b) /
                 (2.0 * weight * (a - b));
        };
        for (int q = 0; q < ext; ++q) {
          const double fq = double(local.f[std::size_t(q)]);
          if (!std::isfinite(fq))
            continue;
          if (m == 0) {
            local.sites[0] = q;
            local.bounds[0] = -std::numeric_limits<double>::infinity();
            local.bounds[1] = std::numeric_limits<double>::infinity();
            m = 1;
            continue;
          }
          int k = m - 1;
          double s = sect(q, local.sites[std::size_t(k)]);
          while (k > 0 && s <= local.bounds[std::size_t(k)]) {
            --k;
            s = sect(q, local.sites[std::size_t(k)]);
          }
          m = k + 2;
          local.sites[std::size_t(k + 1)] = q;
          local.bounds[std::size_t(k + 1)] = s;
          local.bounds[std::size_t(k + 2)] =
              std::numeric_limits<double>::infinity();
        }
        if (m == 0)
          return;
        int k = 0;
        for (int q = 0; q < ext; ++q) {
          while (k + 1 < m && local.bounds[std::size_t(k + 1)] < q)
            ++k;
          const int p = local.sites[std::size_t(k)];
          const double d = double(q - p);
          field[std::size_t(base + q * str)] =
              T(double(local.f[std::size_t(p)]) + weight * d * d);
        }
      },
      local_t{});
}

/// @brief The three passes over one field.
template <typename T>
void seeded_distance_sweep(tf::buffer<T> &field,
                           const std::array<std::ptrdiff_t, 3> &dims,
                           const std::array<double, 3> &weights) {
  for (int axis = 0; axis < 3; ++axis)
    seeded_distance_sweep_axis(field, dims, axis, weights[std::size_t(axis)]);
}

} // namespace volume_detail
} // namespace tf
