/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/range.hpp"
#include "../tree_like.hpp"
#include "tbb/task_arena.h"
#include "tbb/task_group.h"

namespace tf::spatial::impl {

template <typename Tree0, typename Tree1, typename F, typename F1, typename F2>
struct dual_search_params {
  const Tree0 &tree0;
  const Tree1 &tree1;
  const F &bvs_apply;
  const F1 &apply;
  const F2 &abort;
  mutable bool found = false;
};

template <typename Tree0, typename Tree1, typename F, typename F1, typename F2>
auto dual_search(
    std::size_t id0, std::size_t id1, int depth,
    const dual_search_params<Tree0, Tree1, F, F1, F2> &params) {
  if (params.abort())
    return;

  const auto &nodes0 = params.tree0.nodes();
  const auto &nodes1 = params.tree1.nodes();
  const auto &ids0 = params.tree0.ids();
  const auto &ids1 = params.tree1.ids();

  const auto &node0 = nodes0[id0];
  const auto &node1 = nodes1[id1];
  const auto &data0 = node0.get_data();
  const auto &data1 = node1.get_data();

  if (node0.is_leaf() && node1.is_leaf()) {
    if (params.apply(tf::make_range(ids0.begin() + data0[0], data0[1]),
                     tf::make_range(ids1.begin() + data1[0], data1[1]))) {
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
          dual_search(id0, id1, depth - 1, params);
        });
      else
        dual_search(id0, id1, depth, params);
      return true;
    };
    if (node0.is_leaf()) {
      for (auto n_id1 = data1[0]; n_id1 < data1[0] + data1[1]; ++n_id1)
        if (params.bvs_apply(node0.bv, nodes1[n_id1].bv))
          if (!dispatch(id0, n_id1))
            break;

    } else if (node1.is_leaf()) {
      for (auto n_id0 = data0[0]; n_id0 < data0[0] + data0[1]; ++n_id0)
        if (params.bvs_apply(nodes0[n_id0].bv, node1.bv))
          if (!dispatch(n_id0, id1))
            break;
    } else {
      for (auto n_id0 = data0[0]; n_id0 < data0[0] + data0[1]; ++n_id0)
        for (auto n_id1 = data1[0]; n_id1 < data1[0] + data1[1]; ++n_id1)
          if (params.bvs_apply(nodes0[n_id0].bv, nodes1[n_id1].bv))
            if (!dispatch(n_id0, n_id1))
              goto wait_label;
    }
  wait_label:
    tg.wait();
  }
}

template <typename TreePolicy0, typename TreePolicy1, typename F, typename F1,
          typename F2>
auto dual_search(const tf::tree_like<TreePolicy0> &tree0,
                 const tf::tree_like<TreePolicy1> &tree1, const F &bvs_apply,
                 const F1 &apply, const F2 &abort,
                 int parallelism_depth = 6) -> bool {
  const auto &nodes0 = tree0.nodes();
  const auto &nodes1 = tree1.nodes();
  if (!nodes0.size() || !nodes1.size())
    return false;
  if (!bvs_apply(nodes0[0].bv, nodes1[0].bv))
    return false;
  dual_search_params<tf::tree_like<TreePolicy0>, tf::tree_like<TreePolicy1>, F, F1, F2>
      params{tree0, tree1, bvs_apply, apply, abort};
  tbb::task_arena ta;
  ta.execute([&] { dual_search(0, 0, parallelism_depth, params); });
  return params.found;
}

} // namespace tf::spatial::impl
