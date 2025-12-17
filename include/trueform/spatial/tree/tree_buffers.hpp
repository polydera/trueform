/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/aabb.hpp"
#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "./tree_node.hpp"

namespace tf::spatial {

template <typename Index, typename BV> struct tree_buffers {
  using index_type = Index;
  using bv_type = BV;
  using coordinate_type = typename BV::coordinate_type;
  using coordinate_dims = typename BV::coordinate_dims;
  using aabb_type = tf::aabb<coordinate_type, coordinate_dims::value>;
  using node_type = tree_node<Index, BV>;

  tree_buffers() = default;

  auto primitive_aabbs() const { return tf::make_range(_primitive_aabbs); }
  auto primitive_aabbs() { return tf::make_range(_primitive_aabbs); }
  auto nodes() const { return tf::make_range(_nodes); }
  auto nodes() { return tf::make_range(_nodes); }
  auto ids() const { return tf::make_range(_ids); }
  auto ids() { return tf::make_range(_ids); }

protected:
  tf::buffer<aabb_type> _primitive_aabbs;
  tf::buffer<node_type> _nodes;
  tf::buffer<Index> _ids;
};

} // namespace tf::spatial
