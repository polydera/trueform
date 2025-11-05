/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./range.hpp"
#include <type_traits>

namespace tf {
template <typename Policy> struct faces : Policy {
  faces(const Policy &r) : Policy{r} {}
  faces(Policy &&r) : Policy{std::move(r)} {}
};

template <typename Policy>
auto unwrap(const faces<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy> auto unwrap(faces<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy> auto unwrap(faces<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const faces<Policy> &, T &&t) {
  return faces<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(faces<Policy> &, T &&t) {
  return faces<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T> auto wrap_like(faces<Policy> &&, T &&t) {
  return faces<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Range> auto make_faces(Range &&r) {
  auto r0 = tf::make_range(r);
  return tf::faces<decltype(r0)>{r0};
}

template <typename Range> auto make_faces(faces<Range> r) -> faces<Range> {
  return r;
}
template <typename Policy> auto make_view(const tf::faces<Policy> &obj) {
  return obj;
}
} // namespace tf
