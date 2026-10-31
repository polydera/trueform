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
#include <cstddef>

namespace tf::linalg {

/// Add a child's update into the front of its parent: the child's lower
/// triangle over `size` rows, packed by columns, its rows named by
/// `child_rows`, lands where `position` places each row in the front.
///
/// The front stands as in @ref tf::linalg::factor_front_columns. The child's
/// rows ascend and so do their places, so its lower triangle lands in the
/// front's, and every entry takes one addition: the columns go in parallel.
template <typename Index>
auto extend_add_update(const double *update, const Index *child_rows,
                       std::size_t size, const tf::buffer<Index> &position,
                       tf::buffer<Index> &relative, double *head, double *tail,
                       std::size_t m, std::size_t k) -> void {
  relative.allocate(size);
  for (std::size_t a = 0; a < size; ++a)
    relative[a] = position[std::size_t(child_rows[a])];
  const std::size_t rest = m - k;
  tf::parallel_for(
      std::size_t(0), size,
      [&](std::size_t first, std::size_t last) {
        for (std::size_t a = first; a < last; ++a) {
          const double *column = update + a * size - a * (a - 1) / 2 - a;
          const std::size_t to = std::size_t(relative[a]);
          if (to < k) {
            double *into = head + to * m;
            for (std::size_t b = a; b < size; ++b)
              into[std::size_t(relative[b])] += column[b];
          } else {
            double *into = tail + (to - k) * rest;
            for (std::size_t b = a; b < size; ++b)
              into[std::size_t(relative[b]) - k] += column[b];
          }
        }
      },
      tf::checked(256));
}

} // namespace tf::linalg
