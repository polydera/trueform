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
          typename F1, typename Result>
auto tree_proximity(
    const buffer<tree_node<Index, RealT, N>> &nodes, const buffer<Index> &ids,
    const F0 &aabb_metric_f, const F1 &closest_point_f, Result &result) {
  if (!nodes.size())
    return;
  using real_type = RealT;
  struct holder_t {
    real_type metric;
    Index id;

    holder_t(Index id, real_type metric) : metric{metric}, id{id} {};
  };

  tf::small_vector<holder_t, 256> stack;

  auto compare = [](const auto &x, const auto &y) {
    return x.metric > y.metric;
  };

  stack.emplace_back(0, aabb_metric_f(nodes.front().aabb));

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
        auto metric = aabb_metric_f(it->aabb);
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
} // namespace tf::spatial
