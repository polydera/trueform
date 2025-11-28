/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/aabb.hpp"
#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "./tree_node.hpp"
#include <utility>

namespace tf::spatial {

// Owning policy - stores buffers
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

// View policy - stores const pointer ranges
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

// tree_like wrapper
template <typename Policy> struct tree_like : Policy {
  tree_like() = default;
  tree_like(const Policy &policy) : Policy{policy} {}
  tree_like(Policy &&policy) : Policy{std::move(policy)} {}

  using Policy::Policy;
  using Policy::ids;
  using Policy::nodes;
  using Policy::primitive_aabbs;

  using typename Policy::aabb_type;
  using typename Policy::bv_type;
  using typename Policy::coordinate_dims;
  using typename Policy::coordinate_type;
  using typename Policy::index_type;
  using typename Policy::node_type;
};

template <typename Policy>
auto unwrap(const tree_like<Policy> &t) -> decltype(auto) {
  return static_cast<const Policy &>(t);
}

template <typename Policy>
auto unwrap(tree_like<Policy> &t) -> decltype(auto) {
  return static_cast<Policy &>(t);
}

template <typename Policy>
auto unwrap(tree_like<Policy> &&t) -> decltype(auto) {
  return static_cast<Policy &&>(t);
}

template <typename Policy, typename T>
auto wrap_like(const tree_like<Policy> &, T &&t) {
  return tree_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(tree_like<Policy> &, T &&t) {
  return tree_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(tree_like<Policy> &&, T &&t) {
  return tree_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Range> auto make_tree_like(Range &&r) {
  return tree_like<std::decay_t<Range>>{static_cast<Range &&>(r)};
}

// Factory for creating views
template <typename Policy>
auto make_tree_view(const tree_like<Policy> &t) {
  using index_type = typename Policy::index_type;
  using bv_type = typename Policy::bv_type;
  return tree_like<tree_ranges<index_type, bv_type>>{t.nodes(), t.ids(),
                                                      t.primitive_aabbs()};
}

} // namespace tf::spatial
