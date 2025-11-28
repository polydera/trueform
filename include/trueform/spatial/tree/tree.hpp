/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/aabb_from.hpp"
#include "../../core/algorithm/parallel_transform.hpp"
#include "../partitioning.hpp"
#include "./build_aabb_nodes.hpp"
#include "./build_obb_nodes.hpp"
#include "./build_obbrss_nodes.hpp"
#include "./tree_like.hpp"

namespace tf::spatial {

template <typename Index, typename BV>
class tree : public tree_like<tree_buffers<Index, BV>> {
  using base_t = tree_like<tree_buffers<Index, BV>>;

public:
  using base_t::ids;
  using base_t::nodes;
  using base_t::primitive_aabbs;

  using typename base_t::aabb_type;
  using typename base_t::bv_type;
  using typename base_t::coordinate_dims;
  using typename base_t::coordinate_type;
  using typename base_t::index_type;
  using typename base_t::node_type;

  tree() = default;

  template <typename Range>
  tree(const Range &primitives, tree_node_config config) {
    build(primitives, config);
  }

  auto clear() -> void {
    base_t::_primitive_aabbs.clear();
    base_t::_nodes.clear();
    base_t::_ids.clear();
  }

  template <typename Partitioner, typename Range>
  auto build(const Range &primitives, tree_node_config config) -> void {
    base_t::_primitive_aabbs.allocate(primitives.size());
    tf::parallel_transform(primitives, base_t::_primitive_aabbs,
                           [](const auto &x) { return tf::aabb_from(x); });
    build_tree_nodes<Partitioner>(base_t::_nodes, base_t::_ids, primitives,
                                  base_t::_primitive_aabbs, config);
  }

  template <typename Range>
  auto build(const Range &primitives, tree_node_config config) -> void {
    build<nth_element_t>(primitives, config);
  }
};

} // namespace tf::spatial
