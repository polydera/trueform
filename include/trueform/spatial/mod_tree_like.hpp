/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./tree/mod_tree_buffers.hpp"
#include "./tree_like.hpp"
#include <utility>

namespace tf {

template <typename Policy> struct mod_tree_like : Policy {
  mod_tree_like() = default;
  mod_tree_like(const Policy &policy) : Policy{policy} {}
  mod_tree_like(Policy &&policy) : Policy{std::move(policy)} {}

  using Policy::Policy;
  using Policy::delta_ids;
  using Policy::delta_tree;
  using Policy::main_tree;

  using typename Policy::aabb_type;
  using typename Policy::bv_type;
  using typename Policy::coordinate_dims;
  using typename Policy::coordinate_type;
  using typename Policy::index_type;
  using typename Policy::node_type;
};

template <typename Policy>
auto unwrap(const mod_tree_like<Policy> &t) -> decltype(auto) {
  return static_cast<const Policy &>(t);
}

template <typename Policy>
auto unwrap(mod_tree_like<Policy> &t) -> decltype(auto) {
  return static_cast<Policy &>(t);
}

template <typename Policy>
auto unwrap(mod_tree_like<Policy> &&t) -> decltype(auto) {
  return static_cast<Policy &&>(t);
}

template <typename Policy, typename T>
auto wrap_like(const mod_tree_like<Policy> &, T &&t) {
  return mod_tree_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(mod_tree_like<Policy> &, T &&t) {
  return mod_tree_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(mod_tree_like<Policy> &&, T &&t) {
  return mod_tree_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Range> auto make_mod_tree_like(Range &&r) {
  return mod_tree_like<std::decay_t<Range>>{static_cast<Range &&>(r)};
}

// Factory for creating views
template <typename Policy>
auto make_tree_view(const mod_tree_like<Policy> &t) {
  using index_type = typename Policy::index_type;
  using bv_type = typename Policy::bv_type;
  // Create core ranges (nodes + ids only) and pass shared primitive_aabbs
  auto main = t.main_tree();
  auto delta = t.delta_tree();
  return mod_tree_like<spatial::mod_tree_ranges<index_type, bv_type>>{
      spatial::tree_ranges_core<index_type, bv_type>{main.nodes(), main.ids()},
      spatial::tree_ranges_core<index_type, bv_type>{delta.nodes(), delta.ids()},
      t.primitive_aabbs(),
      t.delta_ids()};
}

} // namespace tf
