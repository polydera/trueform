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

/// How many rows each column of the factor holds, its diagonal included, for
/// a matrix whose lower columns and postordered elimination tree are stated.
///
/// Row `i` of the factor is a subtree of the elimination tree whose leaves
/// are among the columns `j` with a lower entry `(i, j)`. A column's count is
/// the number of row subtrees it lies in, summed up the tree as differences:
/// each leaf adds one, and where two leaves of one row meet, at the least
/// common ancestor of the leaf and the row's previous leaf, one is taken
/// back. The ancestors are a union-find with path compression, so the counts
/// cost the matrix's entries rather than the factor's.
template <typename Index>
auto count_cholesky_columns(
    const tf::buffer<Index> &lower_offsets, const tf::buffer<Index> &lower_rows,
    const tf::buffer<Index> &parent, tf::buffer<Index> &first_descendant,
    tf::buffer<Index> &latest_first, tf::buffer<Index> &previous_leaf,
    tf::buffer<Index> &ancestor, tf::buffer<Index> &counts) -> void {
  const std::size_t n = parent.size();
  first_descendant.allocate_and_initialize(n, Index(-1));
  counts.allocate(n);
  for (std::size_t k = 0; k < n; ++k) {
    counts[k] = Index(first_descendant[k] == Index(-1));
    for (Index j = Index(k);
         j != Index(-1) && first_descendant[std::size_t(j)] == Index(-1);
         j = parent[std::size_t(j)])
      first_descendant[std::size_t(j)] = Index(k);
  }
  latest_first.allocate_and_initialize(n, Index(-1));
  previous_leaf.allocate_and_initialize(n, Index(-1));
  ancestor.allocate(n);
  for (std::size_t k = 0; k < n; ++k)
    ancestor[k] = Index(k);
  for (std::size_t j = 0; j < n; ++j) {
    if (parent[j] != Index(-1))
      --counts[std::size_t(parent[j])];
    for (Index e = Index(lower_offsets[j] + 1); e < lower_offsets[j + 1]; ++e) {
      const std::size_t i = std::size_t(lower_rows[std::size_t(e)]);
      if (first_descendant[j] <= latest_first[i])
        continue;
      latest_first[i] = first_descendant[j];
      const Index before = previous_leaf[i];
      previous_leaf[i] = Index(j);
      ++counts[j];
      if (before == Index(-1))
        continue;
      Index meet = before;
      while (meet != ancestor[std::size_t(meet)])
        meet = ancestor[std::size_t(meet)];
      for (Index at = before; at != meet;) {
        const Index up = ancestor[std::size_t(at)];
        ancestor[std::size_t(at)] = meet;
        at = up;
      }
      --counts[std::size_t(meet)];
    }
    if (parent[j] != Index(-1))
      ancestor[j] = parent[j];
  }
  for (std::size_t j = 0; j < n; ++j)
    if (parent[j] != Index(-1))
      counts[std::size_t(parent[j])] =
          Index(counts[std::size_t(parent[j])] + counts[j]);
}

} // namespace tf::linalg
