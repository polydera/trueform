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
#include "../../core/config/verify.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::linalg {

/// The supernodal tree and each supernode's rows: its own columns, then the
/// rows below them ascending, as many as its last column's count states.
///
/// The rows below a supernode are those of its columns' lower entries and of
/// its children's rows below them, each kept once and only past its last
/// column; children come before their parent in a postorder, so one
/// ascending pass reads finished lists.
template <typename Index>
auto make_supernode_rows(
    const tf::buffer<Index> &supernode_offsets, const tf::buffer<Index> &parent,
    const tf::buffer<Index> &counts, const tf::buffer<Index> &lower_offsets,
    const tf::buffer<Index> &lower_rows, tf::buffer<Index> &owner,
    tf::buffer<Index> &head, tf::buffer<Index> &next, tf::buffer<Index> &stamp,
    tf::buffer<Index> &supernode_parent, tf::buffer<Index> &row_offsets,
    tf::buffer<Index> &rows) -> void {
  const std::size_t n = parent.size();
  const std::size_t supernodes = supernode_offsets.size() - 1;
  owner.allocate(n);
  for (std::size_t s = 0; s < supernodes; ++s)
    for (Index j = supernode_offsets[s]; j < supernode_offsets[s + 1]; ++j)
      owner[std::size_t(j)] = Index(s);
  supernode_parent.allocate(supernodes);
  row_offsets.allocate(supernodes + 1);
  head.allocate_and_initialize(supernodes, Index(-1));
  next.allocate(supernodes);
  std::size_t total = 0;
  for (std::size_t s = supernodes; s-- > 0;) {
    const std::size_t last = std::size_t(supernode_offsets[s + 1]) - 1;
    const Index up = parent[last];
    supernode_parent[s] = up == Index(-1) ? up : owner[std::size_t(up)];
    if (up != Index(-1)) {
      next[s] = head[std::size_t(supernode_parent[s])];
      head[std::size_t(supernode_parent[s])] = Index(s);
    }
  }
  for (std::size_t s = 0; s < supernodes; ++s) {
    row_offsets[s] = Index(total);
    const std::size_t last = std::size_t(supernode_offsets[s + 1]) - 1;
    total +=
        last - std::size_t(supernode_offsets[s]) + std::size_t(counts[last]);
  }
  row_offsets[supernodes] = Index(total);

  rows.allocate(total);
  stamp.allocate_and_initialize(n, Index(-1));
  for (std::size_t s = 0; s < supernodes; ++s) {
    const Index first = supernode_offsets[s];
    const Index last = Index(supernode_offsets[s + 1] - 1);
    Index *out = rows.begin() + std::size_t(row_offsets[s]);
    std::size_t at = 0;
    for (Index j = first; j <= last; ++j)
      out[at++] = j;
    const std::size_t below = at;
    auto keep = [&](Index i) {
      if (i > last && stamp[std::size_t(i)] != Index(s)) {
        stamp[std::size_t(i)] = Index(s);
        out[at++] = i;
      }
    };
    for (Index j = first; j <= last; ++j)
      for (Index e = lower_offsets[std::size_t(j)];
           e < lower_offsets[std::size_t(j) + 1]; ++e)
        keep(lower_rows[std::size_t(e)]);
    for (Index child = head[s]; child != Index(-1);
         child = next[std::size_t(child)]) {
      const std::size_t c = std::size_t(child);
      const std::size_t width =
          std::size_t(supernode_offsets[c + 1] - supernode_offsets[c]);
      for (Index p = Index(row_offsets[c] + Index(width));
           p < row_offsets[c + 1]; ++p)
        keep(rows[std::size_t(p)]);
    }
    std::sort(out + below, out + at);
    if constexpr (tf::core::verification)
      tf::core::verify(at == std::size_t(row_offsets[s + 1] - row_offsets[s]),
                       "a supernode holds the rows its last column counts");
  }
}

} // namespace tf::linalg
