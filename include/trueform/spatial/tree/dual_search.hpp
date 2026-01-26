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
#include "../mod_tree_like.hpp"
#include "../tree_like.hpp"
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
                     tf::make_range(ids1.begin() + data1[0], data1[1]),
                     params.tree0.primitive_aabbs(),
                     params.tree1.primitive_aabbs())) {
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
  // Use task_group to ensure calling thread joins the default arena on wait().
  // This guarantees a valid current_thread_index() for local_vector/local_buffer,
  // without creating a separate arena that would have different thread indices.
  tbb::task_group tg;
  tg.run([&] { dual_search(0, 0, parallelism_depth, params); });
  tg.wait();
  return params.found;
}

// mod_tree_like vs tree_like
template <typename ModTreePolicy, typename TreePolicy, typename F, typename F1,
          typename F2>
auto dual_search(const tf::mod_tree_like<ModTreePolicy> &mod_tree,
                 const tf::tree_like<TreePolicy> &tree, const F &bvs_apply,
                 const F1 &apply, const F2 &abort,
                 int parallelism_depth = 6) -> bool {
  bool found0 = false, found1 = false;
  tbb::task_group tg;
  tg.run([&] {
    found0 = dual_search(mod_tree.main_tree(), tree, bvs_apply, apply, abort,
                         parallelism_depth);
  });
  if (mod_tree.delta_tree().ids().size())
    tg.run([&] {
      found1 = dual_search(mod_tree.delta_tree(), tree, bvs_apply, apply, abort,
                           parallelism_depth);
    });
  tg.wait();
  return found0 || found1;
}

// tree_like vs mod_tree_like
template <typename TreePolicy, typename ModTreePolicy, typename F, typename F1,
          typename F2>
auto dual_search(const tf::tree_like<TreePolicy> &tree,
                 const tf::mod_tree_like<ModTreePolicy> &mod_tree,
                 const F &bvs_apply, const F1 &apply, const F2 &abort,
                 int parallelism_depth = 6) -> bool {
  bool found0 = false, found1 = false;
  tbb::task_group tg;
  tg.run([&] {
    found0 = dual_search(tree, mod_tree.main_tree(), bvs_apply, apply, abort,
                         parallelism_depth);
  });
  if (mod_tree.delta_tree().ids().size())
    tg.run([&] {
      found1 = dual_search(tree, mod_tree.delta_tree(), bvs_apply, apply, abort,
                           parallelism_depth);
    });
  tg.wait();
  return found0 || found1;
}

// mod_tree_like vs mod_tree_like
template <typename ModTreePolicy0, typename ModTreePolicy1, typename F,
          typename F1, typename F2>
auto dual_search(const tf::mod_tree_like<ModTreePolicy0> &mod_tree0,
                 const tf::mod_tree_like<ModTreePolicy1> &mod_tree1,
                 const F &bvs_apply, const F1 &apply, const F2 &abort,
                 int parallelism_depth = 6) -> bool {
  bool found0 = false, found1 = false, found2 = false, found3 = false;
  bool has_delta0 = mod_tree0.delta_tree().ids().size();
  bool has_delta1 = mod_tree1.delta_tree().ids().size();
  tbb::task_group tg;
  tg.run([&] {
    found0 = dual_search(mod_tree0.main_tree(), mod_tree1.main_tree(), bvs_apply,
                         apply, abort, parallelism_depth);
  });
  if (has_delta1)
    tg.run([&] {
      found1 = dual_search(mod_tree0.main_tree(), mod_tree1.delta_tree(),
                           bvs_apply, apply, abort, parallelism_depth);
    });
  if (has_delta0)
    tg.run([&] {
      found2 = dual_search(mod_tree0.delta_tree(), mod_tree1.main_tree(),
                           bvs_apply, apply, abort, parallelism_depth);
    });
  if (has_delta0 && has_delta1)
    tg.run([&] {
      found3 = dual_search(mod_tree0.delta_tree(), mod_tree1.delta_tree(),
                           bvs_apply, apply, abort, parallelism_depth);
    });
  tg.wait();
  return found0 || found1 || found2 || found3;
}

} // namespace tf::spatial::impl
