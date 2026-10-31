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

/// The graph a symmetric pattern states: node `i` adjacent to every `j != i`
/// it shares an entry with, each list ascending, and `entries` naming the
/// input entry each adjacency stands for.
///
/// A row's list opens with its own entries below the diagonal, which ascend
/// and lie before it, and closes with the rows below that state it: the
/// lower triangle's transpose, counted, offset by a prefix and scattered in
/// ascending row order through `cursor`.
template <typename Index>
auto make_symmetric_adjacency(const tf::symmetric_lower<Index> &matrix,
                              tf::buffer<Index> &cursor,
                              tf::buffer<Index> &offsets,
                              tf::buffer<Index> &adjacency,
                              tf::buffer<Index> &entries) -> void {
  const std::size_t n = matrix.size();
  offsets.allocate_and_initialize(n + 1, Index(0));
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t first = std::size_t(matrix.row_offsets[i]);
    const std::size_t below = std::size_t(matrix.row_offsets[i + 1]) - 1;
    offsets[i + 1] = Index(offsets[i + 1] + Index(below - first));
    for (std::size_t slot = first; slot < below; ++slot)
      ++offsets[std::size_t(matrix.columns[slot]) + 1];
  }
  for (std::size_t i = 0; i < n; ++i)
    offsets[i + 1] = Index(offsets[i + 1] + offsets[i]);

  adjacency.allocate(std::size_t(offsets[n]));
  entries.allocate(std::size_t(offsets[n]));
  cursor.allocate(n);
  tf::parallel_for(
      std::size_t(0), n,
      [&](std::size_t first, std::size_t last) {
        for (std::size_t i = first; i < last; ++i) {
          const std::size_t below = std::size_t(matrix.row_offsets[i + 1]) - 1;
          std::size_t at = std::size_t(offsets[i]);
          for (std::size_t slot = std::size_t(matrix.row_offsets[i]);
               slot < below; ++slot, ++at) {
            adjacency[at] = matrix.columns[slot];
            entries[at] = Index(slot);
          }
          cursor[i] = Index(at);
        }
      },
      tf::checked);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t below = std::size_t(matrix.row_offsets[i + 1]) - 1;
    for (std::size_t slot = std::size_t(matrix.row_offsets[i]); slot < below;
         ++slot) {
      auto &at = cursor[std::size_t(matrix.columns[slot])];
      adjacency[std::size_t(at)] = Index(i);
      entries[std::size_t(at)] = Index(slot);
      at = Index(at + 1);
    }
  }
}

} // namespace tf::linalg
