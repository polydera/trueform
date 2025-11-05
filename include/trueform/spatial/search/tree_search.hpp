/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/small_vector.hpp"
#include "../tree_node.hpp"

namespace tf::spatial {
template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1>
auto tree_search(const buffer<tree_node<Index, RealT, N>> &nodes,
                 const buffer<Index> &ids, const F0 &aabb_check,
                 const F1 &leaf_apply) {
  if (!nodes.size())
    return false;
  tf::small_vector<Index, 512> stack;
  stack.push_back(0);
  while (stack.size()) {
    auto current_i = stack.back();
    stack.pop_back();
    const auto &node = nodes[current_i];
    const auto &data = node.get_data();
    if (node.is_leaf()) {
      if (leaf_apply(tf::make_range(ids.begin() + data[0], data[1])))
        return true;
      continue;
    }
    auto it = nodes.begin() + data[0];
    auto end = it + data[1];
    auto next_id = data[0];
    while (it != end) {
      if (aabb_check(it->aabb))
        stack.push_back(next_id);
      ++it;
      ++next_id;
    }
  }
  return false;
}
} // namespace tf::spatial
