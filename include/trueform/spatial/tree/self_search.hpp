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
#include "../tree_like.hpp"
#include "tbb/task_group.h"

namespace tf::spatial::impl {

template <typename Tree, typename F, typename F1, typename F2>
struct self_search_params {
  const Tree &tree;
  const F &bvs_apply;
  const F1 &apply;
  const F2 &abort;
  mutable bool found = false;
};

template <typename Tree, typename F, typename F1, typename F2>
auto self_search(int id0, int id1, int depth,
                 const self_search_params<Tree, F, F1, F2> &params) {
  if (params.abort())
    return;

  const auto &nodes = params.tree.nodes();
  const auto &ids = params.tree.ids();

  const auto &node0 = nodes[id0];
  const auto &node1 = nodes[id1];
  const auto &data0 = node0.get_data();
  const auto &data1 = node1.get_data();

  if (node0.is_leaf() && node1.is_leaf()) {
    if (params.apply(tf::make_range(ids.begin() + data0[0], data0[1]),
                     tf::make_range(ids.begin() + data1[0], data1[1]),
                     id0 == id1)) {
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
          self_search(id0, id1, depth - 1, params);
        });
      else
        self_search(id0, id1, depth, params);
      return true;
    };
    if (node0.is_leaf()) {
      for (auto n_id1 = data1[0]; n_id1 < data1[0] + data1[1]; ++n_id1)
        if (params.bvs_apply(node0.bv, nodes[n_id1].bv))
          if (!dispatch(id0, n_id1))
            break;

    } else if (node1.is_leaf()) {
      for (auto n_id0 = data0[0]; n_id0 < data0[0] + data0[1]; ++n_id0)
        if (params.bvs_apply(nodes[n_id0].bv, node1.bv))
          if (!dispatch(n_id0, id1))
            break;
    } else {
      for (std::decay_t<decltype(data0[0])> i0 = 0; i0 < data0[1]; ++i0) {
        auto n_id0 = data0[0] + i0;
        for (std::decay_t<decltype(data1[0])> i1 = i0 * (id0 == id1);
             i1 < data1[1]; ++i1) {
          auto n_id1 = data1[0] + i1;
          if (n_id0 == n_id1 ||
              params.bvs_apply(nodes[n_id0].bv, nodes[n_id1].bv))
            if (!dispatch(n_id0, n_id1))
              goto wait_label;
        }
      }
    }
  wait_label:
    tg.wait();
  }
}

template <typename TreePolicy, typename F, typename F1, typename F2>
auto self_search(const tf::tree_like<TreePolicy> &tree, const F &bvs_apply,
                 const F1 &apply, const F2 &abort,
                 int parallelism_depth = 6) -> bool {
  const auto &nodes = tree.nodes();
  if (!nodes.size())
    return false;
  self_search_params<tf::tree_like<TreePolicy>, F, F1, F2> params{
      tree, bvs_apply, apply, abort};
  tbb::task_group tg;
  tg.run([&] { self_search(0, 0, parallelism_depth, params); });
  tg.wait();
  return params.found;
}

} // namespace tf::spatial::impl
