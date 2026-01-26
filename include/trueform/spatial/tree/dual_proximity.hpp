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
#include "../../core/small_vector.hpp"
#include "./local_tree_metric_result.hpp"
#include "../mod_tree_like.hpp"
#include "../tree_like.hpp"
#include "tbb/task_group.h"
#include <utility>

namespace tf::spatial::impl {

template <typename Tree0, typename Tree1, typename F0, typename F1, typename T>
struct dual_proximity_params {
  const Tree0 &tree0;
  const Tree1 &tree1;
  const F0 &bv_dists_f;
  const F1 &closest_pts;
  tf::spatial::local_tree_metric_result<T> &result;
};

template <typename Tree0, typename Tree1, typename F0, typename F1, typename T>
void dual_proximity_parallel(
    typename Tree0::index_type id0, typename Tree1::index_type id1,
    const dual_proximity_params<Tree0, Tree1, F0, F1, T> &params) {
  using Index0 = typename Tree0::index_type;
  using Index1 = typename Tree1::index_type;
  using RealT = typename Tree0::coordinate_type;

  // References to tree data
  const auto nodes0 = params.tree0.nodes();
  const auto nodes1 = params.tree1.nodes();
  const auto ids0 = params.tree0.ids();
  const auto ids1 = params.tree1.ids();
  const auto &node0 = nodes0[id0];
  const auto &node1 = nodes1[id1];
  const auto &data0 = node0.get_data();
  const auto &data1 = node1.get_data();

  // Child-pair structure
  struct holder_t {
    Index0 id0;
    Index1 id1;
  };
  tf::small_vector<holder_t, 16> children;
  std::pair<RealT, RealT> best_bv{std::numeric_limits<RealT>::max(),
                                  std::numeric_limits<RealT>::max()};
  int best_id = 0;

  // Helper to push viable child pairs
  auto push_if_viable = [&](Index0 c0, Index1 c1) {
    auto [dmin_c, dmax_c] = params.bv_dists_f(nodes0[c0].bv, nodes1[c1].bv);
    if (!params.result.reject_bvs(dmin_c)) {
      params.result.update_bv_max(dmax_c);
      auto current_bv = std::make_pair(dmin_c, dmax_c);
      if (current_bv < best_bv) {
        best_bv = current_bv;
        best_id = children.size();
      }
      children.push_back({c0, c1});
    }
  };

  // Gather children based on leaf status
  if (!node0.is_leaf() && !node1.is_leaf()) {
    for (auto c0 = data0[0]; c0 < data0[0] + data0[1]; ++c0)
      for (auto c1 = data1[0]; c1 < data1[0] + data1[1]; ++c1)
        push_if_viable(c0, c1);
  } else if (!node0.is_leaf()) {
    for (auto c0 = data0[0]; c0 < data0[0] + data0[1]; ++c0)
      push_if_viable(c0, id1);
  } else if (!node1.is_leaf()) {
    for (auto c1 = data1[0]; c1 < data1[0] + data1[1]; ++c1)
      push_if_viable(id0, c1);
  } else {
    // Leaf-leaf: do direct point-to-point checks
    for (auto c0 = data0[0]; c0 < data0[0] + data0[1]; ++c0) {
      for (auto c1 = data1[0]; c1 < data1[0] + data1[1]; ++c1) {
        if (params.result.update(std::make_pair(ids0[c0], ids1[c1]),
                                 params.closest_pts(ids0[c0], ids1[c1])))
          return;
      }
    }
    return;
  }

  // No viable children
  if (children.empty())
    return;

  // Find best child and swap to front
  std::swap(children[0], children[best_id]);

  // Inline descend into best child
  dual_proximity_parallel(children[0].id0, children[0].id1, params);

  // Spawn remaining branches in parallel
  tbb::task_group tg;
  for (size_t i = 1; i < children.size(); ++i) {
    auto ch = children[i];
    tg.run([&, ch] { dual_proximity_parallel(ch.id0, ch.id1, params); });
  }
  tg.wait();
}

template <typename Tree0, typename Tree1, typename F0, typename F1, typename T>
auto dual_proximity_pre_pass(
    const dual_proximity_params<Tree0, Tree1, F0, F1, T> &params,
    int dispatch_depth) {
  using Index0 = typename Tree0::index_type;
  using Index1 = typename Tree1::index_type;
  using RealT = typename Tree0::coordinate_type;

  struct holder_t {
    RealT min2;
    RealT min_max2;
    Index0 id0;
    Index1 id1;
    int depth;
  };

  const auto nodes0 = params.tree0.nodes();
  const auto nodes1 = params.tree1.nodes();
  const auto ids0 = params.tree0.ids();
  const auto ids1 = params.tree1.ids();

  tf::small_vector<holder_t, 256> stack;

  auto push_f = [&](Index0 id0, Index1 id1, int depth) {
    const auto &node0 = nodes0[id0];
    const auto &node1 = nodes1[id1];
    auto [bv_min, bv_max] = params.bv_dists_f(node0.bv, node1.bv);
    if (params.result.reject_bvs(bv_min))
      return;
    params.result.update_bv_max(bv_max);
    stack.push_back({static_cast<RealT>(bv_min), static_cast<RealT>(bv_max),
                     id0, id1, depth});
  };

  auto dispatch = [&](int last_id) {
    std::sort(stack.begin() + std::max(last_id - 4, 0), stack.end(),
              [](const auto &x, const auto &y) {
                return std::make_pair(x.min2, x.min_max2) >
                       std::make_pair(y.min2, y.min_max2);
              });
  };

  push_f(0, 0, 0);
  tbb::task_group tg;
  // so our thread will have a current index in the task_arena
  tg.run([&] {
    while (stack.size()) {
      auto candidate = stack.back();
      stack.pop_back();
      if (params.result.reject_bvs(candidate.min2))
        continue;
      if (candidate.depth > dispatch_depth) {
        tg.run([id0 = candidate.id0, id1 = candidate.id1, &params] {
          dual_proximity_parallel(id0, id1, params);
        });
        continue;
      }
      int last_id = stack.size();
      const auto &node0 = nodes0[candidate.id0];
      const auto &node1 = nodes1[candidate.id1];
      const auto &data0 = node0.get_data();
      const auto &data1 = node1.get_data();

      if (!node0.is_leaf() && !node1.is_leaf()) {
        for (auto n_id0 = data0[0]; n_id0 < data0[0] + data0[1]; ++n_id0)
          for (auto n_id1 = data1[0]; n_id1 < data1[0] + data1[1]; ++n_id1)
            push_f(n_id0, n_id1, candidate.depth + 1);
        dispatch(last_id);
      } else if (!node0.is_leaf()) {
        for (auto n_id0 = data0[0]; n_id0 < data0[0] + data0[1]; ++n_id0)
          push_f(n_id0, candidate.id1, candidate.depth + 1);
        dispatch(last_id);
      } else if (!node1.is_leaf()) {
        for (auto n_id1 = data1[0]; n_id1 < data1[0] + data1[1]; ++n_id1)
          push_f(candidate.id0, n_id1, candidate.depth + 1);
        dispatch(last_id);
      } else {
        for (auto n_id0 = data0[0]; n_id0 < data0[0] + data0[1]; ++n_id0)
          for (auto n_id1 = data1[0]; n_id1 < data1[0] + data1[1]; ++n_id1) {
            if (params.result.update(
                    std::make_pair(ids0[n_id0], ids1[n_id1]),
                    params.closest_pts(ids0[n_id0], ids1[n_id1])))
              return;
          }
      }
    }
  });
  tg.wait();
}

template <typename TreePolicy0, typename TreePolicy1, typename F0, typename F1,
          typename T>
auto dual_proximity(const tf::tree_like<TreePolicy0> &tree0,
                    const tf::tree_like<TreePolicy1> &tree1, const F0 &bv_dists_f,
                    const F1 &closest_pts, tf::spatial::local_tree_metric_result<T> &result,
                    int dispatch_depth = 4) {
  dual_proximity_params<tf::tree_like<TreePolicy0>, tf::tree_like<TreePolicy1>, F0, F1,
                        T>
      params{tree0, tree1, bv_dists_f, closest_pts, result};
  dual_proximity_pre_pass(params, dispatch_depth);
}

// mod_tree_like vs tree_like
template <typename ModTreePolicy, typename TreePolicy, typename F0, typename F1,
          typename T>
auto dual_proximity(const tf::mod_tree_like<ModTreePolicy> &mod_tree,
                    const tf::tree_like<TreePolicy> &tree,
                    const F0 &bv_metrics_f, const F1 &closest_points_f,
                    tf::spatial::local_tree_metric_result<T> &result,
                    int dispatch_depth = 4) {
  tbb::task_group tg;
  tg.run([&] {
    dual_proximity(mod_tree.main_tree(), tree, bv_metrics_f, closest_points_f,
                   result, dispatch_depth);
  });
  if (mod_tree.delta_tree().ids().size())
    tg.run([&] {
      dual_proximity(mod_tree.delta_tree(), tree, bv_metrics_f, closest_points_f,
                     result, dispatch_depth);
    });
  tg.wait();
}

// tree_like vs mod_tree_like
template <typename TreePolicy, typename ModTreePolicy, typename F0, typename F1,
          typename T>
auto dual_proximity(const tf::tree_like<TreePolicy> &tree,
                    const tf::mod_tree_like<ModTreePolicy> &mod_tree,
                    const F0 &bv_metrics_f, const F1 &closest_points_f,
                    tf::spatial::local_tree_metric_result<T> &result,
                    int dispatch_depth = 4) {
  tbb::task_group tg;
  tg.run([&] {
    dual_proximity(tree, mod_tree.main_tree(), bv_metrics_f, closest_points_f,
                   result, dispatch_depth);
  });
  if (mod_tree.delta_tree().ids().size())
    tg.run([&] {
      dual_proximity(tree, mod_tree.delta_tree(), bv_metrics_f, closest_points_f,
                     result, dispatch_depth);
    });
  tg.wait();
}

// mod_tree_like vs mod_tree_like (4-way parallel)
template <typename ModTreePolicy0, typename ModTreePolicy1, typename F0,
          typename F1, typename T>
auto dual_proximity(const tf::mod_tree_like<ModTreePolicy0> &mod_tree0,
                    const tf::mod_tree_like<ModTreePolicy1> &mod_tree1,
                    const F0 &bv_metrics_f, const F1 &closest_points_f,
                    tf::spatial::local_tree_metric_result<T> &result,
                    int dispatch_depth = 4) {
  bool has_delta0 = mod_tree0.delta_tree().ids().size();
  bool has_delta1 = mod_tree1.delta_tree().ids().size();

  tbb::task_group tg;
  tg.run([&] {
    dual_proximity(mod_tree0.main_tree(), mod_tree1.main_tree(), bv_metrics_f,
                   closest_points_f, result, dispatch_depth);
  });
  if (has_delta1)
    tg.run([&] {
      dual_proximity(mod_tree0.main_tree(), mod_tree1.delta_tree(), bv_metrics_f,
                     closest_points_f, result, dispatch_depth);
    });
  if (has_delta0)
    tg.run([&] {
      dual_proximity(mod_tree0.delta_tree(), mod_tree1.main_tree(), bv_metrics_f,
                     closest_points_f, result, dispatch_depth);
    });
  if (has_delta0 && has_delta1)
    tg.run([&] {
      dual_proximity(mod_tree0.delta_tree(), mod_tree1.delta_tree(),
                     bv_metrics_f, closest_points_f, result, dispatch_depth);
    });
  tg.wait();
}

} // namespace tf::spatial::impl
