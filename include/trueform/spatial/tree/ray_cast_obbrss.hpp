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
#include "../../core/dot.hpp"
#include "../../core/intersect_status.hpp"
#include "../../core/obb_like.hpp"
#include "../../core/obbrss.hpp"
#include "../../core/ray_like.hpp"
#include "../../core/ray_obb_check.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../mod_tree_like.hpp"
#include "../tree_like.hpp"

namespace tf::spatial {

template <typename TreePolicy, typename RayPolicy, typename Result, typename F>
auto ray_cast(
    const tf::tree_like<TreePolicy> &tree,
    const tf::ray_like<TreePolicy::coordinate_dims::value, RayPolicy> &ray,
    Result &result, const F &intersect_f,
    const tf::obbrss<typename TreePolicy::coordinate_type,
                     TreePolicy::coordinate_dims::value> &) {
  using Index = typename TreePolicy::index_type;
  using real_t =
      tf::coordinate_type<typename TreePolicy::coordinate_type, RayPolicy>;

  const auto &nodes = tree.nodes();
  const auto &ids = tree.ids();
  if (!nodes.size())
    return;

  tf::small_vector<Index, 256> stack;
  stack.push_back(0);

  real_t t_min = std::numeric_limits<real_t>::max();
  real_t t_max = -std::numeric_limits<real_t>::max();

  auto min_t = result.min_t();
  auto max_t = result.max_t();
  while (stack.size()) {
    auto current_i = stack.back();
    stack.pop_back();
    const auto &node = nodes[current_i];
    const auto &bv = node.bv;
    auto obb = tf::make_obb_like(bv.obb_origin, bv.axes, bv.extent);
    auto hit = core::ray_obb_check(ray, obb, t_min, t_max, min_t, max_t) ==
               tf::intersect_status::intersection;
    if (hit) {
      const auto &data = node.get_data();
      if (!node.is_leaf()) {
        auto nexts = tf::make_sequence_range(data[0], data[0] + data[1]);
        auto dir_dot = tf::dot(ray.direction, bv.axes[0]);
        bool is_negative = dir_dot < real_t(0);
        if (!is_negative) {
          std::reverse_copy(nexts.begin(), nexts.end(),
                            std::back_inserter(stack));
        } else {
          std::copy(nexts.begin(), nexts.end(), std::back_inserter(stack));
        }
      } else {
        for (const auto &id : tf::make_range(ids.begin() + data[0], data[1]))
          max_t = result.update(id, intersect_f(ray, id));
      }
    }
  }
}

// mod_tree_like overload
template <typename ModTreePolicy, typename RayPolicy, typename Result,
          typename F>
auto ray_cast(const tf::mod_tree_like<ModTreePolicy> &tree,
              const tf::ray_like<ModTreePolicy::coordinate_dims::value, RayPolicy>
                  &ray,
              Result &result, const F &intersect_f,
              const tf::obbrss<typename ModTreePolicy::coordinate_type,
                               ModTreePolicy::coordinate_dims::value> &tag) {
  ray_cast(tree.main_tree(), ray, result, intersect_f, tag);
  ray_cast(tree.delta_tree(), ray, result, intersect_f, tag);
}

} // namespace tf::spatial
