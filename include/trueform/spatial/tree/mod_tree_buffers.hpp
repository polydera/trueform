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
#include "./tree_buffers.hpp"
#include "./tree_node.hpp"
#include "./tree_ranges.hpp"
#include "../tree_like.hpp"

namespace tf::spatial {

// =============================================================================
// mod_tree_buffers - owning storage with shared primitive_aabbs
// =============================================================================

template <typename Index, typename BV> struct mod_tree_buffers {
  using index_type = Index;
  using bv_type = BV;
  using coordinate_type = typename BV::coordinate_type;
  using coordinate_dims = typename BV::coordinate_dims;
  using aabb_type = tf::aabb<coordinate_type, coordinate_dims::value>;
  using node_type = tree_node<Index, BV>;

  mod_tree_buffers() = default;

  // Access sub-trees as tree_like views (with shared primitive_aabbs)
  auto main_tree() const {
    return tf::tree_like<tree_ranges<Index, BV>>{
        _main.nodes(), _main.ids(), tf::make_range(_primitive_aabbs)};
  }
  auto delta_tree() const {
    return tf::tree_like<tree_ranges<Index, BV>>{
        _delta.nodes(), _delta.ids(), tf::make_range(_primitive_aabbs)};
  }

  // Access delta_ids as range
  auto delta_ids() const { return tf::make_range(_delta_ids); }

  // Shared primitive_aabbs access
  auto primitive_aabbs() const { return tf::make_range(_primitive_aabbs); }
  auto primitive_aabbs() { return tf::make_range(_primitive_aabbs); }
  auto primitive_aabbs_buffer() -> tf::buffer<aabb_type> & {
    return _primitive_aabbs;
  }
  auto primitive_aabbs_buffer() const -> const tf::buffer<aabb_type> & {
    return _primitive_aabbs;
  }

  // Direct buffer access for sub-trees (nodes + ids only)
  auto main_tree_buffer() -> tree_buffers_core<Index, BV> & { return _main; }
  auto main_tree_buffer() const -> const tree_buffers_core<Index, BV> & {
    return _main;
  }
  auto delta_tree_buffer() -> tree_buffers_core<Index, BV> & { return _delta; }
  auto delta_tree_buffer() const -> const tree_buffers_core<Index, BV> & {
    return _delta;
  }
  auto delta_ids_buffer() -> tf::buffer<Index> & { return _delta_ids; }
  auto delta_ids_buffer() const -> const tf::buffer<Index> & {
    return _delta_ids;
  }

protected:
  tf::buffer<aabb_type> _primitive_aabbs;  // Shared between main and delta
  tree_buffers_core<Index, BV> _main;       // Nodes + ids only
  tree_buffers_core<Index, BV> _delta;      // Nodes + ids only
  tf::buffer<Index> _delta_ids;
};

// =============================================================================
// mod_tree_ranges - non-owning views with shared primitive_aabbs
// =============================================================================

template <typename Index, typename BV> struct mod_tree_ranges {
  using index_type = Index;
  using bv_type = BV;
  using coordinate_type = typename BV::coordinate_type;
  using coordinate_dims = typename BV::coordinate_dims;
  using aabb_type = tf::aabb<coordinate_type, coordinate_dims::value>;
  using node_type = tree_node<Index, BV>;

  using ids_range_type = tf::range<const Index *, tf::dynamic_size>;
  using aabbs_range_type = tf::range<const aabb_type *, tf::dynamic_size>;

  mod_tree_ranges(tree_ranges_core<Index, BV> main,
                  tree_ranges_core<Index, BV> delta,
                  aabbs_range_type primitive_aabbs,
                  ids_range_type delta_ids)
      : _main{main}, _delta{delta}, _primitive_aabbs{primitive_aabbs},
        _delta_ids{delta_ids} {}

  // Access sub-trees as tree_like views (with shared primitive_aabbs)
  auto main_tree() const {
    return tf::tree_like<tree_ranges<Index, BV>>{
        _main.nodes(), _main.ids(), _primitive_aabbs};
  }
  auto delta_tree() const {
    return tf::tree_like<tree_ranges<Index, BV>>{
        _delta.nodes(), _delta.ids(), _primitive_aabbs};
  }

  // Shared primitive_aabbs access
  auto primitive_aabbs() const { return _primitive_aabbs; }

  // Access delta_ids
  auto delta_ids() const -> const ids_range_type & { return _delta_ids; }

protected:
  tree_ranges_core<Index, BV> _main;
  tree_ranges_core<Index, BV> _delta;
  aabbs_range_type _primitive_aabbs;  // Shared between main and delta
  ids_range_type _delta_ids;
};

} // namespace tf::spatial
