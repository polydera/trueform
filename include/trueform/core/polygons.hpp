/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./base/polygons.hpp"
#include "./base/soup.hpp"
#include "./faces.hpp"
#include "./points.hpp"

namespace tf {

template <typename Policy> struct polygons : Policy {
  polygons(const Policy &r) : Policy{r} {}
  polygons(Policy &&r) : Policy{std::move(r)} {}
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
  auto r0 = tf::make_faces(faces);
  auto r1 = tf::make_points(points);
  return polygons<core::polygons<decltype(r0), decltype(r1)>>{
      core::polygons<decltype(r0), decltype(r1)>{r0, r1}};
}

template <typename Range>
auto make_polygons(polygons<Range> p) -> polygons<Range> {
  return p;
}

template <typename Range> auto make_polygons(Range &&r) {
  auto polys = tf::make_range(r);
  return polygons<core::soup<decltype(polys)>>{
      core::soup<decltype(polys)>{std::move(polys)}};
}

template <typename Policy> auto make_view(const tf::polygons<Policy> &obj) {
  return obj;
}

} // namespace tf
