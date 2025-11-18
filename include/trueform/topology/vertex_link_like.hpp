/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <utility>

namespace tf {
template <typename Policy> struct vertex_link_like : Policy {
  vertex_link_like() = default;
  vertex_link_like(const Policy &policy) : Policy{policy} {}
  vertex_link_like(Policy &&policy) : Policy{std::move(policy)} {}
};

template <typename Policy>
auto unwrap(const vertex_link_like<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy>
auto unwrap(vertex_link_like<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy>
auto unwrap(vertex_link_like<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const vertex_link_like<Policy> &, T &&t) {
  return vertex_link_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(vertex_link_like<Policy> &, T &&t) {
  return vertex_link_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(vertex_link_like<Policy> &&, T &&t) {
  return vertex_link_like<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Range> auto make_vertex_link_like(Range &&r) {
  return tf::vertex_link_like<std::decay_t<Range>>{
      static_cast<Range &&>(r)};
}
} // namespace tf

