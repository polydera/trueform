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
#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "./tree_node.hpp"

namespace tf::spatial {

// Core tree buffers without primitive_aabbs (for use in mod_tree sub-trees)
template <typename Index, typename BV> struct tree_buffers_core {
  using index_type = Index;
  using bv_type = BV;
  using coordinate_type = typename BV::coordinate_type;
  using coordinate_dims = typename BV::coordinate_dims;
  using aabb_type = tf::aabb<coordinate_type, coordinate_dims::value>;
  using node_type = tree_node<Index, BV>;

  tree_buffers_core() = default;

  auto nodes() const { return tf::make_range(_nodes); }
  auto nodes() { return tf::make_range(_nodes); }
  auto ids() const { return tf::make_range(_ids); }
  auto ids() { return tf::make_range(_ids); }

  auto nodes_buffer() -> tf::buffer<node_type> & { return _nodes; }
  auto nodes_buffer() const -> const tf::buffer<node_type> & { return _nodes; }
  auto ids_buffer() -> tf::buffer<Index> & { return _ids; }
  auto ids_buffer() const -> const tf::buffer<Index> & { return _ids; }

  auto clear() {
    _nodes.clear();
    _ids.clear();
  }

protected:
  tf::buffer<node_type> _nodes;
  tf::buffer<Index> _ids;
};

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

  auto primitive_aabbs_buffer() -> tf::buffer<aabb_type> & { return _primitive_aabbs; }
  auto primitive_aabbs_buffer() const -> const tf::buffer<aabb_type> & { return _primitive_aabbs; }
  auto nodes_buffer() -> tf::buffer<node_type> & { return _nodes; }
  auto nodes_buffer() const -> const tf::buffer<node_type> & { return _nodes; }
  auto ids_buffer() -> tf::buffer<Index> & { return _ids; }
  auto ids_buffer() const -> const tf::buffer<Index> & { return _ids; }

protected:
  tf::buffer<aabb_type> _primitive_aabbs;
  tf::buffer<node_type> _nodes;
  tf::buffer<Index> _ids;
};

} // namespace tf::spatial
