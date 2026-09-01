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
#include "../../core/ray_like.hpp"
#include "../mod_tree_like.hpp"
#include "../tree_like.hpp"
#include <array>
#include <cmath>
#include <limits>

namespace tf::spatial {

// Per-ray slab state, built once per traversal. The octant is baked into
// flat-float indices of the box (min at [0, Dims), max at [Dims, 2*Dims))
// so side selection is a register offset, not a per-box branch. Bounds
// are computed subtract-before-multiply: with an axis-parallel ray the
// inverse is huge and (a - b) * inv stays exact at box boundaries where
// a * inv - b * inv cancels. A degenerate axis yields 0 * inf = NaN,
// which fmax/fmin discard per IEEE, leaving the interval open along
// that axis.
template <std::size_t Dims, typename RealT> struct ray_slab {
  RealT origin[Dims];
  RealT inv[Dims];
  unsigned lo_ix[Dims];
  unsigned hi_ix[Dims];

  template <typename Ray> auto init(const Ray &ray) -> void {
    for (std::size_t i = 0; i < Dims; ++i) {
      origin[i] = RealT(ray.origin[i]);
      inv[i] = tf::epsilon_inverse(ray.direction[i]);
      const auto n = inv[i] < RealT(0);
      lo_ix[i] = unsigned(n ? Dims + i : i);
      hi_ix[i] = unsigned(n ? i : Dims + i);
    }
  }

  auto neg(int axis) const -> bool { return lo_ix[axis] >= Dims; }

  template <typename BV>
  auto operator()(const BV &bv, RealT min_t, RealT max_t) const -> bool {
    static_assert(sizeof(BV) == 2 * Dims * sizeof(RealT),
                  "slab indexing assumes a flat min/max coordinate layout");
    const auto *b = reinterpret_cast<const RealT *>(&bv);
    RealT lo[Dims], hi[Dims];
    for (std::size_t i = 0; i < Dims; ++i) {
      lo[i] = (b[lo_ix[i]] - origin[i]) * inv[i];
      hi[i] = (b[hi_ix[i]] - origin[i]) * inv[i];
    }
    RealT t0, t1;
    if constexpr (Dims == 3) {
      t0 = std::fmax(std::fmax(lo[0], lo[1]), std::fmax(lo[2], min_t));
      t1 = std::fmin(std::fmin(hi[0], hi[1]), hi[2]);
    } else {
      t0 = min_t;
      t1 = std::numeric_limits<RealT>::max();
      for (std::size_t i = 0; i < Dims; ++i) {
        t0 = std::fmax(lo[i], t0);
        t1 = std::fmin(hi[i], t1);
      }
    }
    t1 += std::abs(t1) * (RealT(2) * std::numeric_limits<RealT>::epsilon());
    return t0 <= std::fmin(t1, max_t);
  }
};

template <typename TreePolicy, typename RayPolicy, typename Result, typename F>
auto ray_cast(
    const tf::tree_like<TreePolicy> &tree,
    const tf::ray_like<TreePolicy::coordinate_dims::value, RayPolicy> &ray,
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

  ray_slab<Dims, real_t> slab;
  slab.init(ray);

  // depth is logarithmic by the balanced build's rank splits, so the
  // worst case fits with a wide margin
  std::array<Index, 256> stack;
  Index sp = 0;
  stack[sp++] = 0;

  const auto min_t = result.min_t();
  auto max_t = result.max_t();
  while (sp) {
    const auto &node = nodes[stack[--sp]];
    if (!slab(node.bv, min_t, max_t))
      continue;
    const auto &data = node.get_data();
    if (!node.is_leaf()) {
      // near child pushed last: popped first, shrinking max_t earliest
      if (!slab.neg(node.axis))
        for (Index c = data[1]; c > 0; --c)
          stack[sp++] = data[0] + c - 1;
      else
        for (Index c = 0; c < data[1]; ++c)
          stack[sp++] = data[0] + c;
    } else {
      for (Index k = 0; k < data[1]; ++k) {
        const auto id = ids[data[0] + k];
        auto info = intersect_f(ray, id);
        if (info)
          max_t = result.update(id, info);
      }
    }
  }
}

// mod_tree_like overload - ray cast against main tree, then delta tree
template <typename ModTreePolicy, typename RayPolicy, typename Result,
          typename F>
auto ray_cast(
    const tf::mod_tree_like<ModTreePolicy> &tree,
    const tf::ray_like<ModTreePolicy::coordinate_dims::value, RayPolicy> &ray,
    Result &result, const F &intersect_f,
    const tf::aabb<typename ModTreePolicy::coordinate_type,
                   ModTreePolicy::coordinate_dims::value> &tag) {
  ray_cast(tree.main_tree(), ray, result, intersect_f, tag);
  ray_cast(tree.delta_tree(), ray, result, intersect_f, tag);
}

} // namespace tf::spatial
