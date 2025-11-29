/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./tree_buffers.hpp"
#include "./tree_ranges.hpp"
#include <utility>

namespace tf {

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
  return tree_like<spatial::tree_ranges<index_type, bv_type>>{
      t.nodes(), t.ids(), t.primitive_aabbs()};
}

} // namespace tf
