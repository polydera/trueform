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
#include "../symmetric_lower.hpp"
#include <cstddef>

namespace tf::linalg {

/// The matrix under the elimination order as the pattern's lower carrier, its
/// columns in the order @ref tf::cholesky_pattern states, and `slot[e]` where
/// input entry `e` stands in it.
///
/// Column `k` gathers the row eliminated `k`-th and the neighbours its graph
/// `adjacency` states that are eliminated after it, each with the entry
/// `entries` names: the columns count, take a prefix, and fill apart.
template <typename Index>
auto make_permuted_lower(const tf::symmetric_lower<Index> &matrix,
                         const tf::buffer<Index> &adjacency_offsets,
                         const tf::buffer<Index> &adjacency,
                         const tf::buffer<Index> &entries,
                         const tf::buffer<Index> &order,
                         const tf::buffer<Index> &rank,
                         tf::buffer<Index> &offsets, tf::buffer<Index> &rows,
                         tf::buffer<Index> &slot) -> void {
  const std::size_t n = matrix.size();
  offsets.allocate(n + 1);
  offsets[0] = Index(0);
  tf::parallel_for(
      std::size_t(0), n,
      [&](std::size_t first, std::size_t last) {
        for (std::size_t k = first; k < last; ++k) {
          const std::size_t row = std::size_t(order[k]);
          Index count = 1;
          for (Index at = adjacency_offsets[row]; at < adjacency_offsets[row + 1];
               ++at)
            if (std::size_t(rank[std::size_t(adjacency[std::size_t(at)])]) > k)
              ++count;
          offsets[k + 1] = count;
        }
      },
      tf::checked);
  for (std::size_t k = 0; k < n; ++k)
    offsets[k + 1] = Index(offsets[k + 1] + offsets[k]);

  rows.allocate(std::size_t(offsets[n]));
  slot.allocate(std::size_t(offsets[n]));
  tf::parallel_for(
      std::size_t(0), n,
      [&](std::size_t first, std::size_t last) {
        for (std::size_t k = first; k < last; ++k) {
          const std::size_t row = std::size_t(order[k]);
          std::size_t position = std::size_t(offsets[k]);
          rows[position] = Index(k);
          slot[std::size_t(matrix.row_offsets[row + 1] - 1)] = Index(position);
          for (Index at = adjacency_offsets[row]; at < adjacency_offsets[row + 1];
               ++at) {
            const Index i = rank[std::size_t(adjacency[std::size_t(at)])];
            if (std::size_t(i) <= k)
              continue;
            rows[++position] = i;
            slot[std::size_t(entries[std::size_t(at)])] = Index(position);
          }
        }
      },
      tf::checked);
}

} // namespace tf::linalg
