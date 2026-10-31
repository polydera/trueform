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

#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/meta.hpp"
#include <array>
#include <cstddef>

namespace tf::intersect::graph {

/// THE BAND JOIN over a position sort. `positions` ascends lexicographically
/// by `point_of(row)`, so a row meets every row within `window` of it on
/// every axis by looking forward alone: the x window is the run past it, and
/// inside each x group the y window and inside each y group the z window are
/// found by search. The cost is the pairs plus one search per group the window
/// reaches, and a row whose successor stands past the x window costs one
/// compare. `pairs` receives `{i, j}`, `i < j`, for every two rows at distinct
/// points within the window, in no order.
template <typename Index, typename Positions, typename PointOf, typename Int>
auto find_near_positions(const Positions &positions, const PointOf &point_of,
                         Int window, tf::buffer<std::array<Index, 2>> &pairs)
    -> void {
  using wide_t = typename tf::exact::meta<Int>::T1;
  const auto n = std::size_t(positions.size());
  const auto first = positions.begin();
  const auto at = [&](std::size_t row, std::size_t k) {
    return wide_t(point_of(first[row])[k]);
  };
  const auto w = wide_t(window);
  // the first row of `[begin, end)` that fails `holds`, which the order makes
  // a prefix of the run
  const auto partition = [](std::size_t begin, std::size_t end,
                            const auto &holds) {
    while (begin < end) {
      const auto mid = begin + (end - begin) / 2;
      if (holds(mid))
        begin = mid + 1;
      else
        end = mid;
    }
    return begin;
  };
  tf::generic_generate(
      tf::make_sequence_range(n), pairs,
      [&](std::size_t i, tf::buffer<std::array<Index, 2>> &out) {
        const auto x = at(i, 0);
        const auto y = at(i, 1);
        const auto z = at(i, 2);
        auto j = i + 1;
        if (j == n || at(j, 0) - x > w)
          return;
        // one y group `[begin, end)`: its z window
        const auto emit_z = [&](std::size_t begin, std::size_t end) {
          for (auto k = partition(
                   begin, end, [&](std::size_t r) { return at(r, 2) < z - w; });
               k < end && at(k, 2) <= z + w; ++k)
            out.push_back({Index(i), Index(k)});
        };
        // the y groups of one x group from `begin` on, up to the window's top
        const auto emit_y = [&](std::size_t begin, std::size_t end) {
          while (begin < end && at(begin, 1) <= y + w) {
            const auto group_y = at(begin, 1);
            const auto group_end =
                partition(begin + 1, end,
                          [&](std::size_t r) { return at(r, 1) == group_y; });
            emit_z(begin, group_end);
            begin = group_end;
          }
        };
        if (at(j, 0) == x) {
          const auto x_end =
              partition(j, n, [&](std::size_t r) { return at(r, 0) == x; });
          // the row's own y group past it ascends in z, and an equal z is the
          // row's own point
          for (; j < x_end && at(j, 1) == y && at(j, 2) - z <= w; ++j)
            if (at(j, 2) != z)
              out.push_back({Index(i), Index(j)});
          if (j < x_end && at(j, 1) == y)
            j = partition(j, x_end,
                          [&](std::size_t r) { return at(r, 1) == y; });
          emit_y(j, x_end);
          j = x_end;
        }
        while (j < n && at(j, 0) - x <= w) {
          const auto group_x = at(j, 0);
          const auto x_end = partition(
              j + 1, n, [&](std::size_t r) { return at(r, 0) == group_x; });
          emit_y(partition(j, x_end,
                           [&](std::size_t r) { return at(r, 1) < y - w; }),
                 x_end);
          j = x_end;
        }
      },
      tf::checked);
}

} // namespace tf::intersect::graph
