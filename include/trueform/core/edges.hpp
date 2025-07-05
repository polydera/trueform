/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./range.hpp"
#include <type_traits>

namespace tf {
template <typename Policy> struct edges : Policy {
  edges(const Policy &r) : Policy{r} {}
  edges(Policy &&r) : Policy{std::move(r)} {}
};

template <typename Policy>
auto unwrap(const edges<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy> auto unwrap(edges<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy> auto unwrap(edges<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const edges<Policy> &, T &&t) {
  return edges<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(edges<Policy> &, T &&t) {
  return edges<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(edges<Policy> &&, T &&t) {
  return edges<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Range> auto make_edges(Range &&r) {
  auto r0 = tf::make_range(r);
  return tf::edges<decltype(r0)>{r0};
}

template <typename Range> auto make_edges(edges<Range> r) -> edges<Range> {
  return r;
}
template <typename Policy> auto make_view(const tf::edges<Policy> &obj) {
  return obj;
}
} // namespace tf
