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
#include "../tree_like.hpp"

namespace tf::spatial::impl {

template <typename TreePolicy, typename F0, typename F1, typename Result>
auto proximity(const tf::tree_like<TreePolicy> &tree, const F0 &bv_metric_f,
               const F1 &closest_point_f, Result &result) {
  using Index = typename TreePolicy::index_type;
  using real_t = typename TreePolicy::coordinate_type;

  const auto &nodes = tree.nodes();
  const auto &ids = tree.ids();
  if (!nodes.size())
    return;

  struct holder_t {
    real_t metric;
    Index id;

    holder_t(Index id, real_t metric) : metric{metric}, id{id} {};
  };

  tf::small_vector<holder_t, 256> stack;

  auto compare = [](const auto &x, const auto &y) {
    return x.metric > y.metric;
  };

  stack.emplace_back(0, bv_metric_f(nodes.front().bv));

  while (stack.size()) {
    auto current = stack.back();
    stack.pop_back();
    if (current.metric > result.metric()) {
      continue;
    }
    const auto &node = nodes[current.id];
    const auto &data = node.get_data();
    if (!node.is_leaf()) {
      auto current_offset = stack.size();
      auto it = nodes.begin() + data[0];
      auto end = it + data[1];
      auto next_id = data[0];
      while (it != end) {
        auto metric = bv_metric_f(it->bv);
        if (metric <= result.metric()) {
          stack.emplace_back(next_id, metric);
        }
        ++it;
        ++next_id;
      }
      std::sort(stack.begin() +
                    std::max(Index(current_offset) - data[1], Index(0)),
                stack.end(), compare);
    } else {
      for (const auto &id : tf::make_range(ids.begin() + data[0], data[1])) {
        auto closest_pt = closest_point_f(id);
        result.update(id, closest_pt);
      }
    }
  }
}

} // namespace tf::spatial::impl
