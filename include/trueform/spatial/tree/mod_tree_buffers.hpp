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
#include "./tree_buffers.hpp"
#include "./tree_node.hpp"
#include "./tree_ranges.hpp"
#include "../tree_like.hpp"

namespace tf::spatial {

// =============================================================================
// mod_tree_buffers - owning storage
// =============================================================================

template <typename Index, typename BV> struct mod_tree_buffers {
  using index_type = Index;
  using bv_type = BV;
  using coordinate_type = typename BV::coordinate_type;
  using coordinate_dims = typename BV::coordinate_dims;
  using aabb_type = tf::aabb<coordinate_type, coordinate_dims::value>;
  using node_type = tree_node<Index, BV>;

  mod_tree_buffers() = default;

  // Access sub-trees as tree_like views
  auto main_tree() const {
    return tf::tree_like<tree_ranges<Index, BV>>{_main.nodes(), _main.ids(),
                                                 _main.primitive_aabbs()};
  }
  auto delta_tree() const {
    return tf::tree_like<tree_ranges<Index, BV>>{_delta.nodes(), _delta.ids(),
                                                 _delta.primitive_aabbs()};
  }

  // Access delta_ids as range
  auto delta_ids() const { return tf::make_range(_delta_ids); }

  // Direct buffer access
  auto main_tree_buffer() -> tree_buffers<Index, BV> & { return _main; }
  auto main_tree_buffer() const -> const tree_buffers<Index, BV> & {
    return _main;
  }
  auto delta_tree_buffer() -> tree_buffers<Index, BV> & { return _delta; }
  auto delta_tree_buffer() const -> const tree_buffers<Index, BV> & {
    return _delta;
  }
  auto delta_ids_buffer() -> tf::buffer<Index> & { return _delta_ids; }
  auto delta_ids_buffer() const -> const tf::buffer<Index> & {
    return _delta_ids;
  }

protected:
  tree_buffers<Index, BV> _main;
  tree_buffers<Index, BV> _delta;
  tf::buffer<Index> _delta_ids;
};

// =============================================================================
// mod_tree_ranges - non-owning views
// =============================================================================

template <typename Index, typename BV> struct mod_tree_ranges {
  using index_type = Index;
  using bv_type = BV;
  using coordinate_type = typename BV::coordinate_type;
  using coordinate_dims = typename BV::coordinate_dims;
  using aabb_type = tf::aabb<coordinate_type, coordinate_dims::value>;
  using node_type = tree_node<Index, BV>;

  using ids_range_type = tf::range<const Index *, tf::dynamic_size>;

  mod_tree_ranges(tf::tree_like<tree_ranges<Index, BV>> main,
                  tf::tree_like<tree_ranges<Index, BV>> delta,
                  ids_range_type delta_ids)
      : _main{main}, _delta{delta}, _delta_ids{delta_ids} {}

  // Access sub-trees as tree_like views
  auto main_tree() const -> const tf::tree_like<tree_ranges<Index, BV>> & {
    return _main;
  }
  auto delta_tree() const -> const tf::tree_like<tree_ranges<Index, BV>> & {
    return _delta;
  }

  // Access delta_ids
  auto delta_ids() const -> const ids_range_type & { return _delta_ids; }

protected:
  tf::tree_like<tree_ranges<Index, BV>> _main;
  tf::tree_like<tree_ranges<Index, BV>> _delta;
  ids_range_type _delta_ids;
};

} // namespace tf::spatial
