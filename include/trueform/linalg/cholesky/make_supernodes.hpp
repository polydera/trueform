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
#include "../../core/buffer.hpp"
#include <cstddef>

namespace tf::linalg {

/// Whether a supernode of `columns` columns may hold `zeros` explicit zeros
/// among its `entries`: every small one may, and larger ones a shrinking
/// share, so a join trades a few stored zeros for one wider dense block.
inline auto cholesky_relaxes(double columns, double zeros, double entries)
    -> bool {
  const double share = zeros / entries;
  return columns <= 4.0 || (columns <= 16.0 && share < 0.8) ||
         (columns <= 48.0 && share < 0.1) || share < 0.05;
}

/// Cut a postordered elimination tree's columns into the supernodes the
/// factor stores as dense blocks: `supernode_offsets` partitions the columns.
///
/// A column joins the next when the next is its parent, its only child, and
/// holds one row fewer, which makes the fundamental supernodes. Then, from
/// the root down, a supernode joins the one above it where that one starts
/// right after it and the joined block keeps its explicit zeros within
/// @ref tf::linalg::cholesky_relaxes. Every column of a supernode lies in its
/// last column's subtree, so the block's rows are its own columns and then
/// the last column's rows below them, and the zeros a join adds follow from
/// the counts alone.
template <typename Index>
auto make_supernodes(const tf::buffer<Index> &parent,
                     const tf::buffer<Index> &counts, tf::buffer<Index> &owner,
                     tf::buffer<Index> &fundamental, tf::buffer<Index> &start,
                     tf::buffer<Index> &into, tf::buffer<double> &zeros,
                     tf::buffer<Index> &supernode_offsets) -> void {
  const std::size_t n = parent.size();
  owner.allocate_and_initialize(n, Index(0));
  for (std::size_t j = 0; j < n; ++j)
    if (parent[j] != Index(-1))
      ++owner[std::size_t(parent[j])];
  fundamental.clear();
  for (std::size_t j = 0; j < n; ++j)
    if (!(j && std::size_t(parent[j - 1]) == j &&
          counts[j - 1] == Index(counts[j] + 1) && owner[j] == Index(1)))
      fundamental.push_back(Index(j));
  fundamental.push_back(Index(n));

  const std::size_t s0 = fundamental.size() - 1;
  start.allocate(s0);
  into.allocate(s0);
  zeros.allocate_and_initialize(s0, 0.0);
  for (std::size_t s = 0; s < s0; ++s) {
    start[s] = fundamental[s];
    into[s] = Index(s);
    for (Index j = fundamental[s]; j < fundamental[s + 1]; ++j)
      owner[std::size_t(j)] = Index(s);
  }
  for (std::size_t s = s0; s-- > 0;) {
    const std::size_t last = std::size_t(fundamental[s + 1]) - 1;
    if (parent[last] == Index(-1))
      continue;
    const std::size_t above =
        std::size_t(into[std::size_t(owner[std::size_t(parent[last])])]);
    if (std::size_t(start[above]) != last + 1)
      continue;
    const double columns = double(last + 1 - std::size_t(fundamental[s]));
    const double rows = columns + double(counts[last]) - 1.0;
    const std::size_t above_last = std::size_t(fundamental[above + 1]) - 1;
    const double above_columns = double(above_last + 1 - (last + 1));
    const double above_rows = above_columns + double(counts[above_last]) - 1.0;
    const double joined_columns = columns + above_columns;
    const double joined_rows = columns + above_rows;
    const double joined_zeros = zeros[above] + columns * (joined_rows - rows);
    const double entries = joined_columns * joined_rows -
                           joined_columns * (joined_columns - 1.0) / 2.0;
    if (!tf::linalg::cholesky_relaxes(joined_columns, joined_zeros, entries))
      continue;
    into[s] = Index(above);
    start[above] = fundamental[s];
    zeros[above] = joined_zeros;
  }

  supernode_offsets.clear();
  for (std::size_t s = 0; s < s0; ++s)
    if (into[s] == Index(s))
      supernode_offsets.push_back(start[s]);
  supernode_offsets.push_back(Index(n));
}

} // namespace tf::linalg
