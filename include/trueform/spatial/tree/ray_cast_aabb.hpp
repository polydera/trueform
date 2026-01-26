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
#include "../../core/aabb.hpp"
#include "../../core/epsilon_inverse.hpp"
#include "../../core/intersect_status.hpp"
#include "../../core/ray_aabb_check.hpp"
#include "../../core/ray_like.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../mod_tree_like.hpp"
#include "../tree_like.hpp"
#include <cstdint>

namespace tf::spatial {

template <typename TreePolicy, typename RayPolicy, typename Result, typename F>
auto ray_cast(const tf::tree_like<TreePolicy> &tree,
              const tf::ray_like<TreePolicy::coordinate_dims::value, RayPolicy>
                  &ray,
              Result &result, const F &intersect_f,
              const tf::aabb<typename TreePolicy::coordinate_type,
                             TreePolicy::coordinate_dims::value> &) {
  using Index = typename TreePolicy::index_type;
  using real_t =
      tf::coordinate_type<typename TreePolicy::coordinate_type, RayPolicy>;
  constexpr std::size_t Dims = TreePolicy::coordinate_dims::value;

  const auto &nodes = tree.nodes();
  const auto &ids = tree.ids();
  if (!nodes.size())
    return;

  tf::vector<real_t, Dims> ray_inv_dir;
  tf::small_vector<Index, 256> stack;
  stack.push_back(0);

  std::array<std::int8_t, Dims> dir_sign;
  for (std::size_t i = 0; i < Dims; ++i) {
    dir_sign[i] = ray.direction[i] < 0;
    ray_inv_dir[i] = tf::epsilon_inverse(ray.direction[i]);
  }

  real_t t_min = std::numeric_limits<real_t>::max();
  real_t t_max = -std::numeric_limits<real_t>::max();

  auto min_t = result.min_t();
  auto max_t = result.max_t();
  while (stack.size()) {
    auto current_i = stack.back();
    stack.pop_back();
    const auto &node = nodes[current_i];
    auto hit =
        core::ray_aabb_check(ray, ray_inv_dir, node.bv, t_min, t_max, min_t,
                             max_t) == tf::intersect_status::intersection;
    if (hit) {
      const auto &data = node.get_data();
      if (!node.is_leaf()) {
        auto nexts = tf::make_sequence_range(data[0], data[0] + data[1]);
        auto is_negative = dir_sign[node.axis];
        if (!is_negative) {
          std::reverse_copy(nexts.begin(), nexts.end(),
                            std::back_inserter(stack));
        } else {
          std::copy(nexts.begin(), nexts.end(), std::back_inserter(stack));
        }
      } else {
        const auto &primitive_aabbs = tree.primitive_aabbs();
        for (const auto &id : tf::make_range(ids.begin() + data[0], data[1])) {
          if (core::ray_aabb_check(ray, ray_inv_dir, primitive_aabbs[id], t_min,
                                   t_max, min_t,
                                   max_t) == tf::intersect_status::intersection)
            max_t = result.update(id, intersect_f(ray, id));
        }
      }
    }
  }
}

// mod_tree_like overload - ray cast against main tree, then delta tree
template <typename ModTreePolicy, typename RayPolicy, typename Result,
          typename F>
auto ray_cast(const tf::mod_tree_like<ModTreePolicy> &tree,
              const tf::ray_like<ModTreePolicy::coordinate_dims::value, RayPolicy>
                  &ray,
              Result &result, const F &intersect_f,
              const tf::aabb<typename ModTreePolicy::coordinate_type,
                             ModTreePolicy::coordinate_dims::value> &tag) {
  ray_cast(tree.main_tree(), ray, result, intersect_f, tag);
  ray_cast(tree.delta_tree(), ray, result, intersect_f, tag);
}

} // namespace tf::spatial
