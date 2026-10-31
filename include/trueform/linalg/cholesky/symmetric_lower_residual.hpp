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
#include "../../core/algorithm/parallel_for.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../cholesky_pattern.hpp"
#include "../symmetric_lower.hpp"
#include <cmath>
#include <cstddef>

namespace tf::linalg {

/// `out = b - A x` over `r` interleaved lanes, and beside it `scale =
/// |b| + |A| |x|`, the size each lane's residual is measured against.
///
/// Each row gathers on its own, in parallel: its entries below the diagonal,
/// its diagonal, and then the entries the rows below state into it, through
/// the pattern's graph and its entry tickets — the order a sweep scattering
/// every entry into both of its rows would sum them in.
template <typename Index>
auto symmetric_lower_residual(const tf::symmetric_lower<Index> &matrix,
                              const tf::cholesky_pattern<Index> &pattern,
                              const tf::buffer<double> &b,
                              const tf::buffer<double> &x, std::size_t r,
                              tf::buffer<double> &out,
                              tf::buffer<double> &scale) -> void {
  const std::size_t n = matrix.size();
  out.allocate(n * r);
  scale.allocate(n * r);
  tf::parallel_for(
      std::size_t(0), n,
      [&](std::size_t first, std::size_t last) {
        for (std::size_t i = first; i < last; ++i) {
          const std::size_t own = std::size_t(matrix.row_offsets[i]);
          const std::size_t diagonal =
              std::size_t(matrix.row_offsets[i + 1]) - 1;
          const std::size_t below =
              std::size_t(pattern.adjacency_offsets[i]) + (diagonal - own);
          const std::size_t end = std::size_t(pattern.adjacency_offsets[i + 1]);
          for (std::size_t c = 0; c < r; ++c) {
            double residual = b[i * r + c];
            double size = std::abs(b[i * r + c]);
            for (std::size_t at = own; at <= diagonal; ++at) {
              const std::size_t j = std::size_t(matrix.columns[at]);
              const double value = matrix.values[at];
              residual -= value * x[j * r + c];
              size += std::abs(value * x[j * r + c]);
            }
            for (std::size_t at = below; at < end; ++at) {
              const std::size_t j = std::size_t(pattern.adjacency[at]);
              const double value =
                  matrix.values[std::size_t(pattern.adjacency_entries[at])];
              residual -= value * x[j * r + c];
              size += std::abs(value * x[j * r + c]);
            }
            out[i * r + c] = residual;
            scale[i * r + c] = size;
          }
        }
      },
      tf::checked);
}

} // namespace tf::linalg
