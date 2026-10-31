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

/// A postorder of a forest stated by `parent`: `post[k]` is the node visited
/// `k`-th, every node after its descendants, children in ascending order and
/// roots likewise, so every subtree is a contiguous run.
template <typename Index>
auto postorder_tree(const tf::buffer<Index> &parent, tf::buffer<Index> &head,
                    tf::buffer<Index> &next, tf::buffer<Index> &stack,
                    tf::buffer<Index> &post) -> void {
  const std::size_t n = parent.size();
  head.allocate_and_initialize(n, Index(-1));
  next.allocate(n);
  post.allocate(n);
  for (std::size_t j = n; j-- > 0;) {
    if (parent[j] == Index(-1))
      continue;
    next[j] = head[std::size_t(parent[j])];
    head[std::size_t(parent[j])] = Index(j);
  }
  std::size_t visited = 0;
  stack.clear();
  for (std::size_t root = 0; root < n; ++root) {
    if (parent[root] != Index(-1))
      continue;
    stack.push_back(Index(root));
    while (stack.size()) {
      const std::size_t top = std::size_t(stack.back());
      const Index child = head[top];
      if (child == Index(-1)) {
        stack.pop_back();
        post[visited++] = Index(top);
      } else {
        head[top] = next[std::size_t(child)];
        stack.push_back(child);
      }
    }
  }
}

} // namespace tf::linalg
