/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./points.hpp"
#include "./views/polygons.hpp"

namespace tf {

template <typename Policy> struct polygons : Policy {
  polygons(const Policy &r) : Policy{r} {}
  polygons(Policy &&r) : Policy{r} {}
};

template <typename Policy>
auto unwrap(const polygons<Policy> &seg) -> decltype(auto) {
  return static_cast<const Policy &>(seg);
}

template <typename Policy>
auto unwrap(polygons<Policy> &seg) -> decltype(auto) {
  return static_cast<Policy &>(seg);
}

template <typename Policy>
auto unwrap(polygons<Policy> &&seg) -> decltype(auto) {
  return static_cast<Policy &&>(seg);
}

template <typename Policy, typename T>
auto wrap_like(const polygons<Policy> &, T &&t) {
  return polygons<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(polygons<Policy> &, T &&t) {
  return polygons<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Policy, typename T>
auto wrap_like(polygons<Policy> &&, T &&t) {
  return polygons<std::decay_t<T>>{static_cast<T &&>(t)};
}

template <typename Range0, typename Range1>
auto make_polygons(Range0 &&faces, Range1 &&points) {
  auto r0 = tf::make_range(faces);
  auto r1 = tf::make_points(points);
  return polygons<views::polygons<decltype(r0), decltype(r1)>>{
      views::polygons<decltype(r0), decltype(r1)>{r0, r1}};
}

template <typename Range>
auto make_polygons(polygons<Range> p) -> polygons<Range> {
  return p;
}

template <typename Range>
auto make_polygons(Range &&r) -> polygons<std::decay_t<Range>> {
  return polygons<std::decay_t<Range>>{static_cast<Range &&>(r)};
}

} // namespace tf
