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
#include "../../core/range.hpp"
#include "./tree_node.hpp"

namespace tf::spatial {

// Core tree ranges without primitive_aabbs (for use in mod_tree sub-trees)
template <typename Index, typename BV> struct tree_ranges_core {
  using index_type = Index;
  using bv_type = BV;
  using coordinate_type = typename BV::coordinate_type;
  using coordinate_dims = typename BV::coordinate_dims;
  using aabb_type = tf::aabb<coordinate_type, coordinate_dims::value>;
  using node_type = tree_node<Index, BV>;

  using nodes_range_type = tf::range<const node_type *, tf::dynamic_size>;
  using ids_range_type = tf::range<const Index *, tf::dynamic_size>;

  tree_ranges_core(nodes_range_type nodes, ids_range_type ids)
      : _nodes{nodes}, _ids{ids} {}

  auto nodes() const { return _nodes; }
  auto ids() const { return _ids; }

protected:
  nodes_range_type _nodes;
  ids_range_type _ids;
};

template <typename Index, typename BV> struct tree_ranges {
  using index_type = Index;
  using bv_type = BV;
  using coordinate_type = typename BV::coordinate_type;
  using coordinate_dims = typename BV::coordinate_dims;
  using aabb_type = tf::aabb<coordinate_type, coordinate_dims::value>;
  using node_type = tree_node<Index, BV>;

  using nodes_range_type = tf::range<const node_type *, tf::dynamic_size>;
  using ids_range_type = tf::range<const Index *, tf::dynamic_size>;
  using aabbs_range_type = tf::range<const aabb_type *, tf::dynamic_size>;

  tree_ranges(nodes_range_type nodes, ids_range_type ids,
              aabbs_range_type primitive_aabbs)
      : _nodes{nodes}, _ids{ids}, _primitive_aabbs{primitive_aabbs} {}

  auto primitive_aabbs() const { return _primitive_aabbs; }
  auto nodes() const { return _nodes; }
  auto ids() const { return _ids; }

protected:
  nodes_range_type _nodes;
  ids_range_type _ids;
  aabbs_range_type _primitive_aabbs;
};

} // namespace tf::spatial
