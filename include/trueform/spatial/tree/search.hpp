/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/range.hpp"
#include "../../core/small_vector.hpp"
#include "../tree_like.hpp"

namespace tf::spatial::impl {

template <typename TreePolicy, typename F0, typename F1>
auto search(const tf::tree_like<TreePolicy> &tree, const F0 &bv_check,
            const F1 &leaf_apply) {
  using Index = typename TreePolicy::index_type;

  const auto &nodes = tree.nodes();
  const auto &ids = tree.ids();
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
      if (bv_check(it->bv))
        stack.push_back(next_id);
      ++it;
      ++next_id;
    }
  }
  return false;
}

} // namespace tf::spatial::impl
