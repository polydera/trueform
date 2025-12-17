/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./range.hpp"
#include <type_traits>

namespace tf {
template <typename Policy> struct paths : Policy {
  paths(const Policy &r) : Policy{r} {}
  paths(Policy &&r) : Policy{std::move(r)} {}
};

template <typename Policy>
auto unwrap(const paths<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy> auto unwrap(paths<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy> auto unwrap(paths<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const paths<Policy> &, T &&t) {
  return paths<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(paths<Policy> &, T &&t) {
  return paths<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(paths<Policy> &&, T &&t) {
  return paths<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Range> auto make_paths(Range &&r) {
  auto r0 = tf::make_range(r);
  return tf::paths<decltype(r0)>{r0};
}

template <typename Range> auto make_paths(paths<Range> r) -> paths<Range> {
  return r;
}
template <typename Policy> auto make_view(const tf::paths<Policy> &obj) {
  return obj;
}
} // namespace tf

