/*
* Copyright (c) 2025 XLAB
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
#include "../../core/range.hpp"
#include "../../core/small_vector.hpp"
#include "../mod_tree_like.hpp"
#include "../tree_like.hpp"

namespace tf::spatial::impl {

template <typename TreePolicy, typename F0, typename F1>
auto search(const tf::tree_like<TreePolicy> &tree, const F0 &bv_check,
            const F1 &leaf_apply) {
  using Index = typename TreePolicy::index_type;

  const auto &nodes = tree.nodes();
  const auto &ids = tree.ids();
  const auto &primitive_aabbs = tree.primitive_aabbs();
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
      if (leaf_apply(tf::make_range(ids.begin() + data[0], data[1]),
                     primitive_aabbs))
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

// mod_tree_like overload - search main tree, then delta tree
template <typename ModTreePolicy, typename F0, typename F1>
auto search(const tf::mod_tree_like<ModTreePolicy> &tree, const F0 &bv_check,
            const F1 &leaf_apply) {
  if (tf::spatial::impl::search(tree.main_tree(), bv_check, leaf_apply))
    return true;
  if (tree.delta_tree().ids().size() &&
      tf::spatial::impl::search(tree.delta_tree(), bv_check, leaf_apply))
    return true;
  return false;
}

} // namespace tf::spatial::impl
