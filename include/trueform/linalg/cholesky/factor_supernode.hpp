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
#include "../cholesky_factor.hpp"
#include "../cholesky_pattern.hpp"
#include "./cholesky_front_scratch.hpp"
#include "./extend_add_update.hpp"
#include "./factor_front_columns.hpp"
#include <cstddef>

namespace tf::linalg {

/// Factor supernode `s` on its front, whose children's updates are the top
/// `children` entries of the run's stack, and leave its own update there in
/// their place.
///
/// The front starts from the supernode's lower columns and takes each
/// child's update in ascending order; its columns factor into the
/// supernode's block of `factor`, and the trailing block, packed by columns,
/// is the update its parent takes. Returns false where a pivot is not
/// positive or not finite.
template <typename Index>
auto factor_supernode(const tf::cholesky_pattern<Index> &pattern,
                      const tf::buffer<double> &lower_values, std::size_t s,
                      std::size_t children,
                      tf::linalg::cholesky_front_scratch<Index> &scratch,
                      tf::cholesky_factor &factor) -> bool {
  const std::size_t first = std::size_t(pattern.supernode_offsets[s]);
  const std::size_t k = std::size_t(pattern.supernode_offsets[s + 1]) - first;
  const Index *rows =
      pattern.rows.begin() + std::size_t(pattern.row_offsets[s]);
  const std::size_t m =
      std::size_t(pattern.row_offsets[s + 1] - pattern.row_offsets[s]);
  const std::size_t rest = m - k;
  double *head = factor.values.begin() + std::size_t(pattern.value_offsets[s]);
  scratch.tail.allocate(rest * rest);
  double *tail = scratch.tail.begin();
  auto &position = scratch.position;
  for (std::size_t p = 0; p < m; ++p)
    position[std::size_t(rows[p])] = Index(p);

  tf::parallel_for(
      std::size_t(0), m,
      [&](std::size_t from, std::size_t to) {
        for (std::size_t j = from; j < to; ++j) {
          if (j < k) {
            for (std::size_t i = 0; i < m; ++i)
              head[j * m + i] = 0.0;
            for (Index e = pattern.lower_offsets[first + j];
                 e < pattern.lower_offsets[first + j + 1]; ++e)
              head[j * m + std::size_t(position[std::size_t(
                               pattern.lower_rows[std::size_t(e)])])] =
                  lower_values[std::size_t(e)];
          } else {
            for (std::size_t i = j; i < m; ++i)
              tail[(j - k) * rest + (i - k)] = 0.0;
          }
        }
      },
      tf::checked(256));

  const std::size_t base = scratch.pending.size() - children;
  for (std::size_t c = base; c < scratch.pending.size(); ++c) {
    const std::size_t child = std::size_t(scratch.pending[c]);
    const std::size_t width = std::size_t(pattern.supernode_offsets[child + 1] -
                                          pattern.supernode_offsets[child]);
    const std::size_t size = std::size_t(pattern.row_offsets[child + 1] -
                                         pattern.row_offsets[child]) -
                             width;
    tf::linalg::extend_add_update(
        scratch.stack.begin() + scratch.pending_offsets[c],
        pattern.rows.begin() + std::size_t(pattern.row_offsets[child]) + width,
        size, position, scratch.relative, head, tail, m, k);
  }
  const std::size_t bottom =
      children ? scratch.pending_offsets[base] : scratch.stack.size();
  scratch.pending.erase_till_end(scratch.pending.begin() + base);
  scratch.pending_offsets.erase_till_end(scratch.pending_offsets.begin() +
                                         base);
  scratch.stack.erase_till_end(scratch.stack.begin() + bottom);

  if (!tf::linalg::factor_front_columns(head, tail, m, k))
    return false;
  if (pattern.supernode_parent[s] == Index(-1))
    return true;
  scratch.pending.push_back(Index(s));
  scratch.pending_offsets.push_back(bottom);
  scratch.stack.reallocate(bottom + rest * (rest + 1) / 2);
  double *update = scratch.stack.begin() + bottom;
  tf::parallel_for(
      std::size_t(0), rest,
      [&](std::size_t from, std::size_t to) {
        for (std::size_t a = from; a < to; ++a) {
          double *column = update + a * rest - a * (a - 1) / 2 - a;
          for (std::size_t b = a; b < rest; ++b)
            column[b] = tail[a * rest + b];
        }
      },
      tf::checked(256));
  return true;
}

} // namespace tf::linalg
