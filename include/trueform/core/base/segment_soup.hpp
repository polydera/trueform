/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <utility>

namespace tf::core {

template <typename Policy> struct segment_soup : Policy {
  segment_soup(const Policy &r) : Policy{r} {}
  segment_soup(Policy &&r) : Policy{std::move(r)} {}
};

template <typename Policy>
auto unwrap(const segment_soup<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy>
auto unwrap(segment_soup<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy>
auto unwrap(segment_soup<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const segment_soup<Policy> &, T &&t) {
  return segment_soup<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(segment_soup<Policy> &, T &&t) {
  return segment_soup<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(segment_soup<Policy> &&, T &&t) {
  return segment_soup<std::decay_t<T>>{static_cast<T &&>(t)};
}
} // namespace tf::core
