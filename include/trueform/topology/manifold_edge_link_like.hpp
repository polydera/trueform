/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <utility>

namespace tf {
template <typename Policy> struct manifold_edge_link_like : Policy {
  manifold_edge_link_like() = default;
  manifold_edge_link_like(const Policy &policy) : Policy{policy} {}
  manifold_edge_link_like(Policy &&policy) : Policy{std::move(policy)} {}
};

template <typename Policy>
auto unwrap(const manifold_edge_link_like<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy>
auto unwrap(manifold_edge_link_like<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy>
auto unwrap(manifold_edge_link_like<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const manifold_edge_link_like<Policy> &, T &&t) {
  return manifold_edge_link_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(manifold_edge_link_like<Policy> &, T &&t) {
  return manifold_edge_link_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(manifold_edge_link_like<Policy> &&, T &&t) {
  return manifold_edge_link_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Range> auto make_manifold_edge_link_like(Range &&r) {
  return tf::manifold_edge_link_like<std::decay_t<Range>>{
      static_cast<Range &&>(r)};
}
} // namespace tf
