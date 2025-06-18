/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./points.hpp"
#include "./segment_range.hpp"

namespace tf {

template <typename Policy> struct segments : Policy {
  segments(const Policy &r) : Policy{r} {}
  segments(Policy &&r) : Policy{r} {}
};

template <typename Policy>
auto unwrap(const segments<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy>
auto unwrap(segments<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy>
auto unwrap(segments<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const segments<Policy> &, T &&t) {
  return segments<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(segments<Policy> &, T &&t) {
  return segments<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(segments<Policy> &&, T &&t) {
  return segments<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Range0, typename Range1>
auto make_segments(Range0 &&edges, Range1 &&points) {
  auto r0 = tf::make_range(edges);
  auto r1 = tf::make_points(points);
  return segments<segment_range<decltype(r0), decltype(r1)>>{
      segment_range<decltype(r0), decltype(r1)>{r0, r1}};
}

template <typename Range>
auto make_segments(segments<Range> p) -> segments<Range> {
  return p;
}

template <typename Range>
auto make_segments(Range &&r) -> segments<std::decay_t<Range>> {
  return segments<std::decay_t<Range>>{static_cast<Range &&>(r)};
}

} // namespace tf
