/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/range.hpp"
#include "tbb/task_group.h"

namespace tf::spatial {
template <typename Range0, typename Range1, typename Range2, typename Range3,
          typename F, typename F1, typename F2>
struct tree_dual_search_params {
  const Range0 &nodes0;
  const Range1 &ids0;
  const Range2 &nodes1;
  const Range3 &ids1;
  const F &boxes_apply;
  const F1 &apply;
  const F2 &abort;
  mutable bool found = false;
};

template <typename Range0, typename Range1, typename Range2, typename Range3,
          typename F, typename F1, typename F2>
auto tree_dual_search(
    int id0, int id1, int depth,
    const tree_dual_search_params<Range0, Range1, Range2, Range3, F, F1, F2>
        &params) {
  if (params.abort())
    return;
  const auto &node0 = params.nodes0[id0];
  const auto &node1 = params.nodes1[id1];
  const auto &data0 = node0.get_data();
  const auto &data1 = node1.get_data();

  if (node0.is_leaf() && node1.is_leaf()) {
    if (params.apply(
            tf::make_range(params.ids0.begin() + data0[0], data0[1]),
            tf::make_range(params.ids1.begin() + data1[0], data1[1]))) {
      params.found = true;
      return;
    }

  } else {
    tbb::task_group tg;
    auto dispatch = [&](int id0, int id1) {
      if (params.abort())
        return false;
      if (depth > 0)
        tg.run([&params, id0, id1, depth] {
          tree_dual_search(id0, id1, depth - 1, params);
        });
      else
        tree_dual_search(id0, id1, depth, params);
      return true;
    };
    if (node0.is_leaf()) {
      for (auto n_id1 = data1[0]; n_id1 < data1[0] + data1[1]; ++n_id1)
        if (params.boxes_apply(node0.aabb, params.nodes1[n_id1].aabb))
          if (!dispatch(id0, n_id1))
            break;

    } else if (node1.is_leaf()) {
      for (auto n_id0 = data0[0]; n_id0 < data0[0] + data0[1]; ++n_id0)
        if (params.boxes_apply(params.nodes0[n_id0].aabb, node1.aabb))
          if (!dispatch(n_id0, id1))
            break;
    } else {
      for (auto n_id0 = data0[0]; n_id0 < data0[0] + data0[1]; ++n_id0)
        for (auto n_id1 = data1[0]; n_id1 < data1[0] + data1[1]; ++n_id1)
          if (params.boxes_apply(params.nodes0[n_id0].aabb,
                                 params.nodes1[n_id1].aabb))
            if (!dispatch(n_id0, n_id1))
              goto wait_label;
    }
  wait_label:
    tg.wait();
  }
}

template <typename Range0, typename Range1, typename Range2, typename Range3,
          typename F, typename F1, typename F2>
auto tree_dual_search(const Range0 &nodes0, const Range1 &ids0,
                      const Range2 &nodes1, const Range3 &ids1,
                      const F &boxes_apply, const F1 &apply, const F2 &abort,
                      int paralelism_depth = 6) -> bool {
  if (!nodes0.size() || !nodes1.size())
    return false;
  if (!boxes_apply(nodes0[0].aabb, nodes1[0].aabb))
    return false;
  tree_dual_search_params<Range0, Range1, Range2, Range3, F, F1, F2> params{
      nodes0, ids0, nodes1, ids1, boxes_apply, apply, abort};
  tree_dual_search(0, 0, paralelism_depth, params);
  return params.found;
}
} // namespace tf::implementation
