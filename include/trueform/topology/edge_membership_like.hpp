/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <utility>

namespace tf {
template <typename Policy> struct edge_membership_like : Policy {
  edge_membership_like() = default;
  edge_membership_like(const Policy &policy) : Policy{policy} {}
  edge_membership_like(Policy &&policy) : Policy{std::move(policy)} {}
};

template <typename Policy>
auto unwrap(const edge_membership_like<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy>
auto unwrap(edge_membership_like<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy>
auto unwrap(edge_membership_like<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const edge_membership_like<Policy> &, T &&t) {
  return edge_membership_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(edge_membership_like<Policy> &, T &&t) {
  return edge_membership_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(edge_membership_like<Policy> &&, T &&t) {
  return edge_membership_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Range> auto make_edge_membership_like(Range &&r) {
  return tf::edge_membership_like<std::decay_t<Range>>{
      static_cast<Range &&>(r)};
}
} // namespace tf
