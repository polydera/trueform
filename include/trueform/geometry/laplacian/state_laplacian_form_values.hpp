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
#include "./laplacian_form.hpp"
#include "./laplacian_system.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::geometry {

/// The values of `k_s L + k_b L M^-1 L + tau M` over the pattern
/// @ref tf::geometry::make_laplacian_form_pattern stated for the same free
/// rows, which must bend wherever `bending` is not zero.
///
/// Entry `(i, j)` is `k_b sum_k L_ik L_kj / m_k + k_s L_ij + tau m_i [i = j]`,
/// its sum taken over `k` as a row of `L` states it — the diagonal first, then
/// the columns ascending — and over `j` likewise, so one system and one free
/// set state one form whatever ran it. The rows are independent; each finds
/// its slots in its own sorted columns.
template <typename Index>
auto state_laplacian_form_values(
    const tf::geometry::laplacian_system<Index> &system,
    const tf::buffer<Index> &free_ids, const tf::buffer<Index> &free_of,
    double stiffness, double bending, double tether,
    tf::geometry::laplacian_form<Index> &form) -> void {
  form.free.values.allocate(form.free.columns.size());
  form.fixed_values.allocate(form.fixed_columns.size());
  tf::parallel_for(
      std::size_t(0), free_ids.size(),
      [&](std::size_t first, std::size_t last) {
        for (std::size_t i = first; i < last; ++i) {
          const std::size_t row = std::size_t(free_ids[i]);
          const auto free_first = form.free.columns.begin() +
                                  std::ptrdiff_t(form.free.row_offsets[i]);
          const auto free_last = form.free.columns.begin() +
                                 std::ptrdiff_t(form.free.row_offsets[i + 1]);
          const auto fixed_first = form.fixed_columns.begin() +
                                   std::ptrdiff_t(form.fixed_offsets[i]);
          const auto fixed_last = form.fixed_columns.begin() +
                                  std::ptrdiff_t(form.fixed_offsets[i + 1]);
          for (auto it = free_first; it != free_last; ++it)
            form.free.values[std::size_t(it - form.free.columns.begin())] = 0.0;
          for (auto it = fixed_first; it != fixed_last; ++it)
            form.fixed_values[std::size_t(it - form.fixed_columns.begin())] =
                0.0;
          const auto add = [&](Index column, double value) {
            const Index position = free_of[std::size_t(column)];
            if (std::size_t(position) >= free_ids.size()) {
              const auto at = std::lower_bound(fixed_first, fixed_last, column);
              form.fixed_values[std::size_t(at - form.fixed_columns.begin())] +=
                  value;
            } else if (std::size_t(position) <= i) {
              const auto at = std::lower_bound(free_first, free_last, position);
              form.free.values[std::size_t(at - form.free.columns.begin())] +=
                  value;
            }
          };
          const auto add_row = [&](std::size_t k, double scale) {
            add(Index(k), scale * system.diagonal[k]);
            for (Index at = system.row_offsets[k];
                 at < system.row_offsets[k + 1]; ++at)
              add(system.columns[std::size_t(at)],
                  scale * system.weights[std::size_t(at)]);
          };
          if (bending != 0.0) {
            add_row(row, system.diagonal[row] / system.mass[row]);
            for (Index at = system.row_offsets[row];
                 at < system.row_offsets[row + 1]; ++at) {
              const std::size_t k =
                  std::size_t(system.columns[std::size_t(at)]);
              add_row(k, system.weights[std::size_t(at)] / system.mass[k]);
            }
            for (auto it = free_first; it != free_last; ++it)
              form.free.values[std::size_t(it - form.free.columns.begin())] *=
                  bending;
            for (auto it = fixed_first; it != fixed_last; ++it)
              form.fixed_values[std::size_t(it - form.fixed_columns.begin())] *=
                  bending;
          }
          if (stiffness != 0.0)
            add_row(row, stiffness);
          if (tether != 0.0)
            add(Index(row), tether * system.mass[row]);
        }
      },
      tf::checked);
}

} // namespace tf::geometry
